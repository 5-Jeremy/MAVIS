import torch, os, time, argparse, atexit, yaml
import numpy as np
import anytree as at
from utils.at_utils import ValNodeTokensHDF5, check_tree
from h5py import File
from utils.loading_utils import load_for_anthropic, load_for_summary, load_for_safeRLHF, set_seed, load_value_model, get_dataset_from_objective
from utils.gen_utils import add_generation_args, add_mavis_args, get_mavis_alg, ValueGeneratorBatched
from utils.hdf5_utils import append_pickled_object

def gen_tree_for_prompt(prompt, tokenizer, mavis, max_completion_len, args):
    device = args["device"] if "device" in args.keys() else "cuda" if torch.cuda.is_available() else "cpu"
    assert ("root_children" in args.keys()) and ("non_root_children" in args.keys()) and ("num_layers" in args.keys())
    K_root = args["root_children"]
    K = args["non_root_children"]
    L = args["num_layers"]

    K_schedule = {0:K_root, 1:K}
    for l in range(2,L):
        K_schedule[l] = K_schedule[l-1]
    
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    # We associate each node with an index into the array within the hdf5 file for that node's tree
    curr_indx = 0
    # Maintain a list of tensors until it is time to save to a hdf5 file
    tree_tokens = []
    # All generated samples can be represented as a tree with the prompt as the root
    root = ValNodeTokensHDF5(curr_indx)
    tree_tokens.append(tokenized_prompt.cpu())
    curr_indx += 1
    # Each element in the incomplete list is a tuple containing the full sequence so far, the current leaf node for that sequence, 
    # and the number of tokens left to generate
    curr_incomplete = [(tokenized_prompt, root, max_completion_len)]
    next_incomplete = []
    collected_samples = []
    for l in range(L):
        for seq, node, tokens_left in curr_incomplete:
            num_children = K_schedule[l]
            # Attention masks shouldn't be needed since we only batch sequences of the same length
            # attn_masks = seq["attention_mask"].repeat(num_children, 1)
            if l == L - 1:
                tokens_to_add = tokens_left
            else:
                tokens_to_add = np.random.randint(1, 2*int((tokens_left - args.get("tokens_left_adjustments", {}).get(l, 0)) / (L-l))) # Upper limit will be 2*int(adjusted_tokens_left / (L-l))-1
            # Generate next layer of tokens
            # Repeat the sequence for the number of children to generate
            seq_batch = seq.repeat(num_children, 1)
            allow_eos_on_first_token = not (l == 0 and not args.get("allow_eos_on_first_token", False))
            expanded_seq, log_prob_ratios = mavis(seq_batch, tokens_to_add, allow_eos_on_first_token=allow_eos_on_first_token)
            # Get only the newly generated tokens
            new_tokens = expanded_seq[:, seq.shape[1]:]  
            # For each sequence, check if there is an EOS token
            eos_mask = (new_tokens == tokenizer.eos_token_id).any(dim=1)
            # Turn the batched tensor into a list of tensors
            new_tokens_list = torch.split(new_tokens, 1, dim=0)
            # Create the nodes; remove any padding if needed
            for indx, y in enumerate(new_tokens_list):
                y_unpadded = y[y != tokenizer.pad_token_id].unsqueeze(0)
                if eos_mask[indx] and l == 0 and not args["no_dynamic_splitting"] and y_unpadded.shape[1] > 1:
                    # Choose a token at which to split the sequence. All tokens before this token will belong to the child
                    # of the root node, and all tokens after it will belong to a child of that node. Then we will generate
                    # non_root_children additional children for the second layer. Note that this will result in 
                    # non_root_children + 1 children total
                    split_index = np.random.randint(1, y_unpadded.shape[1])
                    child_log_prob_ratio = log_prob_ratios[indx, :split_index].sum().item()
                    child = ValNodeTokensHDF5(curr_indx, parent=node, log_prob_ratio=child_log_prob_ratio)
                    tree_tokens.append(y_unpadded[:, :split_index].cpu())
                    curr_indx += 1
                    second_layer_child_log_prob_ratio = log_prob_ratios[indx, split_index:].sum().item()
                    second_layer_child = ValNodeTokensHDF5(curr_indx, parent=child, log_prob_ratio=second_layer_child_log_prob_ratio)
                    tree_tokens.append(y_unpadded[:, split_index:].cpu())
                    curr_indx += 1
                    collected_samples.append(second_layer_child)
                    next_incomplete.append((torch.cat([seq, y_unpadded[:, :split_index]], dim=1), child, tokens_left - split_index))
                else:
                    child_log_prob_ratio = log_prob_ratios[indx, :].sum().item()
                    child = ValNodeTokensHDF5(curr_indx, parent=node, log_prob_ratio=child_log_prob_ratio)
                    tree_tokens.append(y_unpadded.cpu())
                    curr_indx += 1
                    if l == L - 1 or eos_mask[indx]:
                        collected_samples.append(child)
                    else:
                        next_incomplete.append((torch.cat([seq, y_unpadded], dim=1), child, tokens_left - tokens_to_add))
        if len(next_incomplete) == 0:
            break
        curr_incomplete = next_incomplete
        next_incomplete = []
    if l < 1:
        # If we reached the end of the loop without reaching the second layer from the root, discard the tree
        return None, None, 0
    else:
        return root, tree_tokens, len(collected_samples)

if __name__ == "__main__":
    A = argparse.ArgumentParser()
    A.add_argument("start_prompt", type=int)
    A.add_argument("num_prompts", type=int)
    A.add_argument("objective", type=str, help="The objective of the value model which this data will be used to train")
    A.add_argument("--output_dir", type=str, default=None, help="Directory to save the generated trees and hdf5 file")
    add_generation_args(A, {'dataset': None}) # data_split, temperature, topk, device, no_cache
    add_mavis_args(A) # value_model_dir, value_model_iter, beta, allow_eos_on_first_token
    A.add_argument("--root_children", type=int, default=2, help="Number of children to branch off from the root node")
    A.add_argument("--non_root_children", type=int, default=2, help="The number of children to branch off per layer for all subsequent layers")
    A.add_argument("--num_layers", type=int, default=5, help="The number of layers to produce")
    A.add_argument("--no_dynamic_splitting", action="store_true", help="By default extra children will be generated to guarantee that no children of the root node are leaves. Use this flag to disable that behavior.")
    A.add_argument("--verbose_loop", action="store_true", help="If set, prints out additional details during tree generation.")
    args = A.parse_args()
    assert not args.no_cache, "Use of the cache is required for this script"
    assert args.dataset is None, "The dataset argument is not used since it can be inferred from the objective."
    
    if args.no_dynamic_splitting:
        print("Dynamic splitting is disabled. You should verify that the generated trees have a depth of at least 2.")

    dataset = get_dataset_from_objective(args.objective)
    args.dataset = dataset # Set this to make sure the dataset name gets saved to data_info.yaml

    assert args.start_prompt >= 0
    assert args.num_prompts > 0
    start_prompt = args.start_prompt
    num_prompts = args.num_prompts
    # The number of children to branch off from the root node
    K_root = args.root_children
    # The number of children to branch off per layer for all subsequent layers
    K = args.non_root_children
    # The number of layers to produce
    L = args.num_layers

    device = args.device

    output_dir = args.output_dir if args.output_dir is not None else f"{dataset}_K{K_root}-{K}_L{L}_k{args.topk}"
    os.makedirs(output_dir, exist_ok=True)
    # The file where the tokens will be saved
    tree_tokens_file = File(os.path.join(output_dir, "all_data.hdf5"), "a")
    def close_file():
        tree_tokens_file.close()
    atexit.register(close_file)

    set_seed(16)

    prompt_indices = list(range(start_prompt, start_prompt+num_prompts))
    # Make sure we do not overwrite any trees that have already been generated
    prompt_files = [f for f in os.listdir(output_dir) if (f.endswith(".pkl") and f.startswith("prompt"))]
    existing_prompt_indices = [int(f.split("_")[0].replace("prompt","")) for f in prompt_files]
    prompt_indices = [i for i in prompt_indices if i not in existing_prompt_indices]
    if len(prompt_indices) == 0:
        print("All prompt indices already exist. Exiting.")
        exit()

    value_models = {}
    args.value_model_iter = int(args.value_model_iter) # Assume only a single number was given
    subdir = f"iter_{args.value_model_iter}"
    print(f"Loading value models from {os.path.join(args.value_model_dir, subdir)}")
    objective = args.objective
    objectives = [objective]
    # Reward and value models start in the CPU to avoid OOM
    if dataset == "anthropic":
        assert objective in ["help", "harm", "humor"], "Objective must be one of 'help', 'harm', or 'humor'"
        loaded_assets = load_for_anthropic(
            csv_path="datasets/anthropic/",
            prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
            pre_tokenized=False, base_model_type=args.base_model_type
        )
    elif dataset == "summary":
        assert objective in ["summarization", "faithful"], "Objective must be one of 'summarization' or 'faithful'"
        loaded_assets = load_for_summary(
            csv_path="datasets/summary/",
            prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
            pre_tokenized=False, base_model_type=args.base_model_type
        )
    elif dataset == "safeRLHF":
        assert objective in ["safeRLHF_help", "safeRLHF_harm"], "Objective must be one of 'safeRLHF_help' or 'safeRLHF_harm'"
        loaded_assets = load_for_safeRLHF(
            csv_path="datasets/safeRLHF/",
            prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
            pre_tokenized=False
        )
    else:
        raise ValueError("Invalid dataset_type")
    generative_model = loaded_assets["gen_model"]
    tokenizer = loaded_assets["gen_tokenizer"]
    eos_token_id = loaded_assets["eos_token_id"]
    reward_model = loaded_assets["ORM_model"]
    get_rewards = loaded_assets["get_rewards"]
    inputs = loaded_assets["prompts"]
    max_completion_len = loaded_assets["max_completion_len"]
    for objective in objectives:
        value_models[objective] = load_value_model(checkpoint=os.path.join(args.value_model_dir, subdir, objective), torch_dtype=torch.float32, device=device, 
                                                    tokenizer=tokenizer)
        value_models[objective].eval()
        value_models[objective].config.use_cache = not args.no_cache
    value_tokenizer = tokenizer
    generative_model.to(device)

    # This is for reproducing the results we report; if you use a different num_layers then you will have
    # to decide on how to adjust tokens_left yourself, or just leave it as is
    if args.dataset == "safeRLHF" and args.num_layers == 5:
        # For safeRLHF, we modify the formula for the max number of tokens to add in a layer since
        # max_completion_len is much larger. The values here will be subtracted from tokens_left (which is
        # simply max_completion_len minus the number of tokens generated in previous layers) so that it is
        # as if max_completion_len were 120 for the first layer, 165 for the second layer, etc.
        # The reasoning behind this is that we want the upper limit on the length of the first L-1=4 layers
        # to not be too far from max_completion_len, but we don't want the range of lengths for the first 
        # layer to be too large since we especially want the value model to learn values for early tokens 
        # (especially considering most rollouts for this dataset will not reach max_completion_len).
        args.tokens_left_adjustments = {0: 512-120, 1: 512-165, 2: 512-210, 3: 512-255}
    else:
        args.tokens_left_adjustments = {l: 0 for l in range(L-1)}

    print("Start prompt index: ", start_prompt)
    print("Num prompts: ", num_prompts)
    print("K_root: ", K_root)
    print("K: ", K)
    print("L: ", L)
    print("max_completion_len: ", max_completion_len)
    print("topk: ", args.topk)
    print("beta: ", args.beta)

    # Create a yaml file with the dataset information
    if not os.path.exists(os.path.join(output_dir, "data_info.yaml")):
        with open(os.path.join(output_dir, "data_info.yaml"), "w") as f:
            yaml.dump(vars(args), f)
    
    value_generator = ValueGeneratorBatched(value_models, value_tokenizer, device, obj_weights={objective: 1.0}, dtype=torch.float32)

    # Set max_new_tokens to None since that will be different every time we generate
    mavis = get_mavis_alg(tokenizer, generative_model, value_generator, max_new_tokens=None, args=args, 
                            return_log_prob_ratio=True, force_no_eos_on_first_token=(not args.allow_eos_on_first_token))

    start = time.time_ns()

    for index, prompt in list(zip(prompt_indices, inputs)):
        print("Generating tree for prompt index: ", index)
        if args.verbose_loop:
            print("Full prompt: ", prompt)
        root, tree_tokens, num_samples = gen_tree_for_prompt(prompt, tokenizer, mavis, max_completion_len, args.__dict__)
        if root is None:
            print(f"Skipping prompt {index} due to no valid tree generated.")
            continue
        if args.verbose_loop:
            print("Num samples: ", num_samples)
            print("Num nodes: ", len(list(at.PreOrderIter(root))))

        if not check_tree(root, check_values=False, max_tokens_indx=len(tree_tokens)-1):
            print("ERROR: Tree is invalid")
            exit()

        prompt_name = f"prompt{index}"
        # Before saving to the hdf5 file, we need all of the tensors in tree_tokens to have the same length
        max_len = max(t.shape[1] for t in tree_tokens)
        for i in range(len(tree_tokens)):
            if tree_tokens[i].shape[1] < max_len:
                tree_tokens[i] = torch.nn.functional.pad(tree_tokens[i], (0, max_len - tree_tokens[i].shape[1]), value=tokenizer.pad_token_id)
        tree_tokens = torch.stack(tree_tokens, dim=0)
        # Save the tokens to the hdf5 file
        tree_tokens_file.create_dataset(prompt_name, data=tree_tokens.numpy())
        # Save the tree to use as training data
        append_pickled_object(tree_tokens_file, prompt_name, root, group_path="/trees")

    print("Time taken: ", (time.time_ns()-start)/1e9)