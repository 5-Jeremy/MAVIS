import anytree as at
import torch
import numpy as np
import argparse, pickle, os, yaml
from h5py import File
from os import path
from utils.loading_utils import load_for_anthropic, load_for_summary, load_for_safeRLHF
from utils.at_utils import check_tree, ValNodeTokensHDF5
from utils.hdf5_utils import load_all_pickled_objects, append_pickled_object, copy_tokens_between_hdf5_files
from utils.gen_utils import add_generation_args
import time

def get_leaf_values(root, get_ORM_values, tokenizer=None, tree_tokens=None, batch_size=32, compute_expected_log_prob_ratio=False):
     # Get rewards for any leaf nodes (nodes which either reached an eos token or are at the final layer)
    leaf_nodes = [node for node in at.PreOrderIter(root) if node.is_leaf]
    leaf_nodes_batched = [leaf_nodes[i:min(i+batch_size, len(leaf_nodes))] for i in range(0, len(leaf_nodes), batch_size)]
    # Only check one of the nodes to save time
    assert type(leaf_nodes[0]) is ValNodeTokensHDF5 and tokenizer is not None and tree_tokens is not None, "Leaf nodes must be of type ValNodeTokensHDF5 and a tokenizer and tokens file must be provided for decoding."
    if compute_expected_log_prob_ratio:
        assert leaf_nodes[0].log_prob_ratio is not None, "Log probability ratios must be set on all leaf nodes to compute expected log probability ratios."
    for leaf_batch in leaf_nodes_batched:
        text_batch = []
        for leaf in leaf_batch:
            leaf_tokens_indx_path = leaf.get_tokens_indx_path()
            node_sequences = [tree_tokens[indx] for indx in leaf_tokens_indx_path]
            node_sequences_unpadded = [seq[seq != tokenizer.pad_token_id].unsqueeze(0) for seq in node_sequences]
            leaf_sequence = torch.cat(node_sequences_unpadded, dim=1)
            text_batch.append(tokenizer.decode(leaf_sequence[0], skip_special_tokens=True))
        reward_batch = get_ORM_values(text_batch)
        if type(reward_batch) is dict:
            reward_batch_list = {k: v.flatten().tolist() if isinstance(v, torch.Tensor) else v for k, v in reward_batch.items()}
            for indx, leaf in enumerate(leaf_batch):
                leaf.value = {objective: reward_batch_list[objective][indx] for objective in reward_batch_list.keys()}
                if compute_expected_log_prob_ratio:
                    # The "expected" log probability ratio of a leaf node is just the total for the sequence
                    leaf.expected_log_prob_ratio = sum([node.log_prob_ratio for node in leaf.path[1:]])
        else:
            raise ValueError("ORM returned a value that is not a dictionary. This should not happen.")
    return root

# This function should be called after the ORM has assigned values to all leaf nodes. In addition to calculating the values, it
# computes the expected future log probability ratios
def calculate_values(root, objectives=None, compute_expected_log_prob_ratio=False):
    max_depth = root.height
    all_nodes = [node for node in at.PreOrderIter(root)]
    # We iterate through nodes from the lowest level (max_depth-1) to the highest level (1)
    for d in range(max_depth-1, 0, -1):
        nodes = [node for node in all_nodes if node.depth == d]
        # We remove the nodes we are about to process from the list of all nodes to save time searching the list later
        all_nodes = [node for node in all_nodes if node.depth != d]
        for node in nodes:
            children = node.children
            if len(children) == 0:
                # If the node has no children, it is a leaf node that was already processed
                continue
            node.value = {objective: np.mean([child.value[objective] for child in children]).item() for objective in objectives}
            if compute_expected_log_prob_ratio:
                node.expected_log_prob_ratio = np.mean([child.expected_log_prob_ratio for child in children]).item()
    root.value = {objective: np.mean([child.value[objective] for child in root.children]).item() for objective in objectives}
    if compute_expected_log_prob_ratio:
        root.expected_log_prob_ratio = np.mean([child.expected_log_prob_ratio for child in root.children]).item()
    return root

if __name__ == "__main__":
    start_time = time.time()
    # Perform relabling on the tree root loaded from the given pickle file
    A = argparse.ArgumentParser()
    A.add_argument("hdf5_file", type=str, help="Path to the hdf5 file produced during data generation")
    A.add_argument("objective", type=str, help="Indicates the reward model(s) to use for assigning values to leaf nodes")
    A.add_argument("--check_trees", action="store_true", help="Check the trees for validity after relabeling")
    add_generation_args(A) # This is used to add arguments for the dataset and base model type
    # A.add_argument("--dataset", type=str, default='anthropic', choices=['anthropic', 'summary', 'safeRLHF'], help="The dataset used (only required when objective is 'all' and there is no data_info.yaml file in the directory)")
    A.add_argument("--compute_KL", action="store_true", help="Compute expected log probability ratios for each node so that a KL penalty can be applied during training")
    args = A.parse_args()
    # If the directory given has a data_info.yaml file, use that to infer what ORM to use based on the dataset
    if os.path.exists(os.path.join(os.path.dirname(args.hdf5_file), "data_info.yaml")):
        with open(os.path.join(os.path.dirname(args.hdf5_file), "data_info.yaml"), "r") as f:
            data_info = yaml.safe_load(f)
        dataset = data_info["dataset"]
        print(f"Using dataset {dataset} from data_info.yaml")
    elif args.dataset is not None:
        dataset = args.dataset
    else:
        raise ValueError("Cannot determine which dataset is in use. Use the --dataset flag to specify.")
    
    # Prepare to load trees and tokens from the HDF5 file
    data_file = File(args.hdf5_file, "r")

    # If a directory was given, we will iterate over all of the pkl files in the directory
    roots = load_all_pickled_objects(data_file, group_path="/trees")
    # We output a new hdf5 file with the same tokens but with the relabeled trees
    output_filename = path.join(path.dirname(args.hdf5_file), args.objective + "_labeled.hdf5")
    out_data_file = File(output_filename, "w")
    copy_tokens_between_hdf5_files(data_file, out_data_file)
    if dataset == "anthropic":
        assert args.objective in ["help", "harm", "humor", "all"], "Objective must be one of 'help', 'harm', 'humor', or 'all' for the Anthropic dataset."
        if args.objective == "all":
            objectives = ["help", "harm", "humor"]
        else:
            objectives = [args.objective]
        loaded_assets = load_for_anthropic(rewards=objectives, rm_device="cuda:0", include_gen_model=False, include_inputs=False, base_model_type="llama")
        ORM_model, get_ORM_values, ORM_tokenizer = loaded_assets["ORM_model"], loaded_assets["get_rewards"], loaded_assets["ORM_tokenizer"]
        gen_tokenizer = loaded_assets["gen_tokenizer"]
    elif dataset == "summary":
        assert args.objective in ["summarization", "faithful", "all"], "Objective must be one of 'summarization', 'faithful', or 'all' for the Summary dataset."
        if args.objective == "all":
            objectives = ["summarization", "faithful"]
        else:
            objectives = [args.objective]
        # To reproduce the experiments shown in our paper, the faithfulness reward needs to be rescaled when using the LLaMA-13B base model but not when using the LLaMA-7B base model
        loaded_assets = load_for_summary(rewards=objectives, rm_device="cuda:0", include_gen_model=False, include_inputs=False, base_model_type="llama", rescale=(args.base_model_type=="llama_13b"))
        ORM_model, get_ORM_values, ORM_tokenizer = loaded_assets["ORM_model"], loaded_assets["get_rewards"], loaded_assets["ORM_tokenizer"]
        gen_tokenizer = loaded_assets["gen_tokenizer"]
    elif dataset == "safeRLHF":
        assert args.objective in ["safeRLHF_help", "safeRLHF_harm", "all"], "Objective must be one of 'safeRLHF_help', 'safeRLHF_harm', or 'all' for the safeRLHF dataset."
        if args.objective == "all":
            objectives = ["safeRLHF_help", "safeRLHF_harm"]
        else:
            objectives = [args.objective]
        loaded_assets = load_for_safeRLHF(rewards=objectives, rm_device="cuda:0", include_gen_model=False, include_inputs=False)
        ORM_model, get_ORM_values, ORM_tokenizer = loaded_assets["ORM_model"], loaded_assets["get_rewards"], loaded_assets["ORM_tokenizer"]
        gen_tokenizer = loaded_assets["gen_tokenizer"]
    else:
        raise ValueError("Invalid dataset.")

    all_roots_with_leaf_values = {}
    for name, root in roots.items():
        # Each key is of the form "prompt#"
        root_with_leaf_values = get_leaf_values(root, get_ORM_values, tokenizer=gen_tokenizer, tree_tokens=torch.tensor(np.array(data_file[name])), compute_expected_log_prob_ratio=args.compute_KL)
        all_roots_with_leaf_values[name] = root_with_leaf_values
    for name, root in all_roots_with_leaf_values.items():
        new_root = calculate_values(root, objectives, compute_expected_log_prob_ratio=args.compute_KL)
        if args.check_trees:
            if not check_tree(new_root, objectives=objectives, check_probs=args.compute_KL):
                print("ERROR: Tree is invalid")
                exit()
        append_pickled_object(out_data_file, name, new_root, group_path="/trees")
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")