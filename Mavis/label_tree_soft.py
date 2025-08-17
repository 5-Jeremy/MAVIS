import anytree as at
import torch
import numpy as np
import argparse, pickle, os, yaml
from h5py import File
from os import path
from utils.loading_utils import load_for_anthropic, load_for_summary
from utils.at_utils import check_tree, ValNodeTokensHDF5

def get_leaf_values(root, get_ORM_values, tokenizer=None, tree_tokens=None, batch_size=32):
     # Get rewards for any leaf nodes (nodes which either reached an eos token or are at the final layer)
    leaf_nodes = [node for node in at.PreOrderIter(root) if node.is_leaf]
    leaf_nodes_batched = [leaf_nodes[i:min(i+batch_size, len(leaf_nodes))] for i in range(0, len(leaf_nodes), batch_size)]
    assert type(leaf_nodes[0]) is ValNodeTokensHDF5 and tokenizer is not None and tree_tokens is not None, "Leaf nodes must be of type ValNodeTokensHDF5 and a tokenizer and tokens file must be provided for decoding."
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
                # The "expected" log probability ratio of a leaf node is just the total for the sequence
                leaf.expected_log_prob_ratio = sum([node.log_prob_ratio for node in leaf.path[1:]])
        else:
            raise ValueError("ORM returned a value that is not a dictionary. This should not happen.")
    return root

# This function should be called after the ORM has assigned values to all leaf nodes. In addition to calculating the values, it
# computes the expected future log probability ratios
def calculate_values(root, objectives=None):
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
            node.expected_log_prob_ratio = np.mean([child.expected_log_prob_ratio for child in children]).item()
    root.value = {objective: np.mean([child.value[objective] for child in root.children]).item() for objective in objectives}
    root.expected_log_prob_ratio = np.mean([child.expected_log_prob_ratio for child in root.children]).item()
    return root

if __name__ == "__main__":
    # Perform relabling on the tree root loaded from the given pickle file
    A = argparse.ArgumentParser()
    A.add_argument("root_file", type=str, help="Path to the pickle file containing the root of the tree, or a directory with multiple such files")
    A.add_argument("objective", type=str, help="Indicates the reward model(s) to use for assigning values to leaf nodes")
    A.add_argument("--check_trees", action="store_true", help="Check the trees for validity after relabeling")
    A.add_argument("--dataset", type=str, default='anthropic', choices=['anthropic', 'summary'], help="The dataset used (only required when objective is 'all' and there is no data_info.yaml file in the directory)")
    args = A.parse_args()
    # If the directory given has a data_info.yaml file, use that to infer what ORM to use based on the dataset
    if os.path.exists(os.path.join(os.path.dirname(args.root_file), "data_info.yaml")):
        with open(os.path.join(os.path.dirname(args.root_file), "data_info.yaml"), "r") as f:
            data_info = yaml.safe_load(f)
        dataset = data_info["dataset"]
        print(f"Using dataset {dataset} from data_info.yaml")
    elif args.dataset is not None:
        dataset = args.dataset
    else:
        raise ValueError("Cannot determine which dataset is in use. Use the --dataset flag to specify.")
    # If a directory was given, we will iterate over all of the pkl files in the directory
    files = [path.join(path.dirname(args.root_file),f) for f in os.listdir(args.root_file) if f.endswith(".pkl")] if path.isdir(args.root_file) else [args.root_file]
    # Output relabeled trees to a new directory in the same location as the input file or directory
    output_dir = path.join(path.dirname(args.root_file), args.objective + "_labeled")
    os.makedirs(output_dir, exist_ok=True)
    if dataset == "anthropic":
        assert args.objective in ["help", "harm", "humor", "all"], "Objective must be one of 'help', 'harm', 'humor', or 'all' for the Anthropic dataset."
        if args.objective == "all":
            objectives = ["help", "harm", "humor"]
        else:
            objectives = [args.objective]
        loaded_assets = load_for_anthropic(rewards=objectives, rm_device="cuda:0", include_gen_model=False, include_inputs=False)
        ORM_model, get_ORM_values, ORM_tokenizer = loaded_assets["ORM_model"], loaded_assets["get_rewards"], loaded_assets["ORM_tokenizer"]
        gen_tokenizer = loaded_assets["gen_tokenizer"]
    elif dataset == "summary":
        assert args.objective in ["summarization", "faithful", "all"], "Objective must be one of 'summarization', 'faithful', or 'all' for the Summary dataset."
        if args.objective == "all":
            objectives = ["summarization", "faithful"]
        else:
            objectives = [args.objective]
        loaded_assets = load_for_summary(rewards=objectives, rm_device="cuda:0", include_gen_model=False, include_inputs=False)
        ORM_model, get_ORM_values, ORM_tokenizer = loaded_assets["ORM_model"], loaded_assets["get_rewards"], loaded_assets["ORM_tokenizer"]
        gen_tokenizer = loaded_assets["gen_tokenizer"]
    else:
        raise ValueError("Invalid dataset.")
    
    if "all_tokens.hdf5" in os.listdir(path.dirname(args.root_file)):
        # Prepare to load tokens from the HDF5 file
        tokens_file = path.join(path.dirname(args.root_file), "all_tokens.hdf5")
        tokens_file = File(tokens_file, "r")

    all_roots_with_leaf_values = {}
    for file in files:
        with open(file, "rb") as f:
            root = pickle.load(f)
        # Assuming the filename is "prompt#_tree.pkl"
        dataset_name = os.path.basename(file).split("_")[0]
        output_path = path.join(output_dir, path.split(file)[-1])
        root_with_leaf_values = get_leaf_values(root, get_ORM_values, tokenizer=gen_tokenizer, tree_tokens=torch.tensor(np.array(tokens_file[dataset_name])))
        all_roots_with_leaf_values[output_path] = root_with_leaf_values
    for file, root in all_roots_with_leaf_values.items():
        new_root = calculate_values(root, objectives)
        if args.check_trees:
            if not check_tree(new_root, objectives=objectives, check_probs=True):
                print("ERROR: Tree is invalid")
                exit()
        with open(file, "wb") as f:
            pickle.dump(new_root, f)