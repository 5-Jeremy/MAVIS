import random, torch
import numpy as np
import anytree as at
from utils.at_utils import ValNodeTokensHDF5, check_tree
import os, time, yaml, pickle, argparse
from h5py import File
from utils.loading_utils import load_for_anthropic, load_for_summary, load_for_safeRLHF, set_seed
from utils.gen_utils import add_generation_args
from utils.hdf5_utils import append_pickled_object

A = argparse.ArgumentParser()
A.add_argument("start_prompt", type=int)
A.add_argument("num_prompts", type=int)
A.add_argument("--output_dir", type=str, default=None, help="Directory to save the generated trees and hdf5 file")
add_generation_args(A) # dataset, data_split, temperature, topk, device, no_cache
A.add_argument("--root_children", type=int, default=4, help="Number of children to branch off from the root node")
A.add_argument("--non_root_children", type=int, default=2, help="The number of children to branch off per layer for all subsequent layers")
A.add_argument("--num_layers", type=int, default=5, help="The number of layers to produce")
A.add_argument("--no_dynamic_splitting", action="store_true", help="By default extra children will be generated to guarantee that no children of the root node are leaves. Use this flag to disable that behavior.")
A.add_argument("--silent_loop", action="store_true", help="If set, does not print out progress within the main loop")
args = A.parse_args()

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
# The sampling temperature (use 1.0 to match the Almost Surely Safe Alignment paper)
temp = args.temperature
device = args.device

K_schedule = {0:K_root, 1:K}
for l in range(2,L):
    K_schedule[l] = K_schedule[l-1]

output_dir = args.output_dir if args.output_dir is not None else f"Training_Output/{args.dataset}/unlabeled_data/iter0_K{K_root}-{K}_L{L}_k{args.topk}"
os.makedirs(output_dir, exist_ok=True)
# The file where the tokens will be saved
tree_tokens_file = File(os.path.join(output_dir, "all_data.hdf5"), "a")

set_seed(16)

prompt_indices = list(range(start_prompt, start_prompt+num_prompts))
# Make sure we do not overwrite any trees that have already been generated
prompt_files = [f for f in os.listdir(output_dir) if (f.endswith(".pkl") and f.startswith("prompt"))]
existing_prompt_indices = [int(f.split("_")[0].replace("prompt","")) for f in prompt_files]
prompt_indices = [i for i in prompt_indices if i not in existing_prompt_indices]
if len(prompt_indices) == 0:
    print("All prompt indices already exist. Exiting.")
    exit()

# Get the prompts and generative model
if args.dataset == "anthropic":
    loaded_assets = load_for_anthropic(csv_path="datasets/anthropic/", prompt_indices=prompt_indices, split=args.data_split, 
                                       pre_tokenized=False, include_rewards=False, base_model_type=args.base_model_type) 
elif args.dataset == "summary":
    loaded_assets = load_for_summary(csv_path="datasets/summary/", prompt_indices=prompt_indices, split=args.data_split, 
                                       pre_tokenized=False, include_rewards=False, base_model_type=args.base_model_type)
elif args.dataset == "safeRLHF":
    loaded_assets = load_for_safeRLHF(csv_path="datasets/safeRLHF/", prompt_indices=prompt_indices, split=args.data_split,
                                     pre_tokenized=False, include_rewards=False)
else:
    raise ValueError("Invalid dataset choice")
inputs = loaded_assets["prompts"]
generative_model = loaded_assets["gen_model"]
tokenizer = loaded_assets["gen_tokenizer"]
eos_token_id = loaded_assets["eos_token_id"]
max_completion_len = loaded_assets["max_completion_len"]

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
    tokens_left_adjustments = {0: 512-120, 1: 512-165, 2: 512-210, 3: 512-255}
else:
    tokens_left_adjustments = {l: 0 for l in range(L-1)}

print("Start prompt index: ", start_prompt)
print("Num prompts: ", num_prompts)
print("K_root: ", K_root)
print("K: ", K)
print("L: ", L)
print("max_completion_len: ", max_completion_len)

# If this dataset is just being created, we will create a yaml file with the dataset information
# If data is being added on to an existing dataset, check to make sure that the configs match (this is to ensure only data with
# the same parameters is in the same directory to avoid confusion)
if not os.path.exists(os.path.join(output_dir, "data_info.yaml")):
    with open(os.path.join(output_dir, "data_info.yaml"), "w") as f:
        yaml.dump(vars(args), f)
else:
    notice_string = f" - the data that already exists in {output_dir} was created with different parameters, so that directory should be moved or renamed."
    # We do not require num_completions to match
    with open(os.path.join(output_dir, "data_info.yaml"), "r") as f:
        data_info = yaml.safe_load(f)
    assert data_info["dataset"] == args.dataset, "Dataset mismatch" + notice_string
    assert data_info["data_split"] == args.data_split, "Data split mismatch"
    assert data_info["root_children"] == args.root_children, "Root children mismatch" + notice_string
    assert data_info["non_root_children"] == args.non_root_children, "Non-root children mismatch" + notice_string
    assert data_info["num_layers"] == args.num_layers, "Num layers mismatch" + notice_string
    assert data_info["temperature"] == args.temperature, "Temperature mismatch" + notice_string

generative_model.to(device)

def generate_batched(seq, num_tokens, num_traj, bs=25):
    i = 0
    generated_sequences = []
    eos_mask = torch.zeros(num_traj, dtype=torch.bool)
    while i < num_traj:
        with torch.no_grad():
            output = generative_model.generate(seq.repeat(min(bs,num_traj-i), 1), do_sample=True, top_k=args.topk,
                                                 temperature=temp, max_new_tokens=num_tokens, pad_token_id=tokenizer.pad_token_id,
                                                 return_dict_in_generate=True)
        expanded = output["sequences"]
        new_tokens = expanded[:, seq.shape[1]:]
        new_tokens_list = torch.split(new_tokens, 1, dim=0)
        generated_sequences.extend(new_tokens_list)
        # Any sequences which have reached an eos token should not be expanded further
        batch_eos_mask = (expanded == eos_token_id)[:,seq.shape[1]:].sum(dim=1) > 0
        eos_mask[i:i+min(bs,num_traj-i)] = batch_eos_mask
        i += len(generated_sequences)
    return generated_sequences, eos_mask

start = time.time_ns()
num_generated_tokens = 0

for index, prompt in list(zip(prompt_indices, inputs)):
    if not args.silent_loop:
        print("Generating tree for prompt: ", prompt)
        print("Prompt index: ", index)
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
        if not args.silent_loop:
            print("Begin layer: ", l)
        for seq, node, tokens_left in curr_incomplete:
            num_children = K_schedule[l]
            # Attention masks shouldn't be needed since we only batch sequences of the same length
            if l == L - 1:
                tokens_to_add = tokens_left
            else:
                tokens_to_add = np.random.randint(1, 2*int((tokens_left - tokens_left_adjustments[l]) / (L-l))) # Upper limit will be 2*int(adjusted_tokens_left / (L-l))-1
            # Generate next layer of tokens
            expanded_seq, eos_mask = generate_batched(seq, tokens_to_add, num_children)
            # Create nodes; node that each element in expanded_seq contains only newly generated tokens
            for indx, y in enumerate(expanded_seq):
                # First remove all pad tokens
                y = y[y != tokenizer.pad_token_id].unsqueeze(0)
                if eos_mask[indx] and l == 0 and not args.no_dynamic_splitting and y.shape[1] > 1:
                    # Choose a token at which to split the sequence. All tokens before this token will belong to the child
                    # of the root node, and all tokens after it will belong to a child of that node. Then we will generate
                    # non_root_children additional children for the second layer. Note that this will result in 
                    # non_root_children + 1 children total
                    split_index = np.random.randint(1, y.shape[1])
                    child = ValNodeTokensHDF5(curr_indx, parent=node)
                    tree_tokens.append(y[:, :split_index].cpu())
                    curr_indx += 1
                    second_layer_child = ValNodeTokensHDF5(curr_indx, parent=child)
                    tree_tokens.append(y[:, split_index:].cpu())
                    curr_indx += 1
                    collected_samples.append(second_layer_child)
                    next_incomplete.append((torch.cat([seq, y[:, :split_index]], dim=1), child, tokens_left - split_index))
                else:
                    child = ValNodeTokensHDF5(curr_indx, parent=node)
                    tree_tokens.append(y.cpu())
                    num_generated_tokens += y.shape[1]
                    curr_indx += 1
                    if l == L - 1 or eos_mask[indx]:
                        collected_samples.append(child)
                    else:
                        next_incomplete.append((torch.cat([seq, y], dim=1), child, tokens_left - tokens_to_add))
        if len(next_incomplete) == 0:
            break
        curr_incomplete = next_incomplete
        next_incomplete = []
    if l < 1:
        # If we reached the end of the loop without reaching the second layer from the root, discard the tree
        continue
    if not args.silent_loop:
        print("Num samples: ", len(collected_samples))
        print("Num nodes: ", len(list(at.PreOrderIter(root))))

    if not check_tree(root, check_values=False):
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

tree_tokens_file.close()
print("Time taken: ", (time.time_ns()-start)/1e9)
# Create a file in the directory where the data was saved that contains the total number of generated tokens and the total time taken
with open(os.path.join(output_dir, "data_stats.txt"), "w") as f:
    f.write(f"Total number of generated tokens: {num_generated_tokens}\n")
    f.write(f"Total time taken (seconds): {(time.time_ns()-start)/1e9}\n")