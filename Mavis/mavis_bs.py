import numpy as np
import torch
import os, random, yaml
from datetime import datetime
import argparse

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)

from utils.loading_utils import load_for_anthropic, load_for_summary, load_value_model
from utils.search_utils import assign_gpus, eval_loop, parse_objective_weights, parse_value_model_iter
from utils.gen_utils import ValueGeneratorBatched, add_generation_args, add_mavis_args, top_k_from_logits, normalize_values

A = argparse.ArgumentParser()
add_generation_args(A, {"data_split":"test"}) # dataset, data_split, temperature, topk, device, no_cache
add_mavis_args(A, {"normalize_values":True}) # value_model_dir, value_model_iter, beta, normalize_values
A.add_argument("--num_prompts", type=int, default=100, help="Number of prompts to evaluate on; the prompts are selected in order starting from index 0")
A.add_argument("--obj_weights", type=str, default="1.0,0.0,0.0", help="Comma-separated list of weights for the three objectives (Help, Harm, Humor for the anthropic dataset). If you do not want to evaluate on an objective, leave it empty (e.g. 0.5,,0.5).")
A.add_argument("--num_trials", type=int, default=3)
A.add_argument("--track_time", action="store_true")
A.add_argument("--num_samples", type=int, default=1) 
A.add_argument("--Q", type=int, default=3, help="Beam width for beam search")
A.add_argument("--N", type=int, default=None, help="Max number of final candidates to evaluate with the reward models; defaults to the beam width (Q)")
A.add_argument("--log_dir", type=str, default="mavis_bs_logs/")
args = A.parse_args()

verbose = True
prompt_indices = list(range(args.num_prompts))

print("Dataset:", args.dataset)
print(f"Prompt range: {prompt_indices[0]} to {prompt_indices[-1]}")
print("Number of samples (N):", args.num_samples)
print(f"Beam width (Q): {args.Q}")
assert args.Q > 1, "Beam width (Q) should be greater than 1 for beam search; otherwise use mavis.py"

g_device, rm_device = assign_gpus()

if args.dataset == "anthropic":
    dataset_valid_objectives = ["help", "harm", "humor"]
elif args.dataset == "summary":
    dataset_valid_objectives = ["summarization", "faithful"]
else:
    raise ValueError("Invalid dataset_type")

# Parse the given objective weights and value model iteration numbers
obj_weights = parse_objective_weights(args.obj_weights, dataset_valid_objectives)
iter_nums = parse_value_model_iter(args.value_model_iter, dataset_valid_objectives)

objectives = [obj for obj, weight in obj_weights.items() if weight is not None]
value_model_types = [obj for obj, weight in obj_weights.items() if weight is not None and weight > 0.0]

value_models = {}
if args.dataset == "anthropic":
    loaded_assets = load_for_anthropic(
        csv_path="datasets/anthropic/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights
    )
elif args.dataset == "summary":
    loaded_assets = load_for_summary(
        csv_path="datasets/summary/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights
    )
else:
    raise ValueError("Invalid dataset_type")
generative_model = loaded_assets["gen_model"].to(g_device)
tokenizer = loaded_assets["gen_tokenizer"]
eos_token_id = loaded_assets["eos_token_id"]
reward_model = loaded_assets["ORM_model"]
get_rewards = loaded_assets["get_rewards"]
inputs = loaded_assets["prompts"]
max_completion_len = loaded_assets["max_completion_len"]
for objective in value_model_types:
    subdir = f"iter_{iter_nums[objective]}"
    print(f"Loading {objective} value model from {os.path.join(args.value_model_dir, subdir)}")
    print("WARNING: loading value models in float32 precision")
    value_models[objective] = load_value_model(checkpoint=os.path.join(args.value_model_dir, subdir, objective), torch_dtype=torch.float32, device=g_device, 
                                                tokenizer=tokenizer)
    value_models[objective].eval()
    value_models[objective].config.use_cache = True
value_tokenizer = tokenizer

value_generator = ValueGeneratorBatched(value_models, value_tokenizer, g_device, obj_weights, dtype=torch.float32)

def mavis_beam(prompt_str):
    """
    The main differences between this implementation and the basic MAVIS implementation are:
    - Maintaining multiple beams with diverging sequences rather than a single sequence
    - Scoring across beams with unnormalized values
    """
    # Initialize beam with prompt
    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(g_device)
    beams = [input_ids]  # Start with prompt
    
    beam_width = args.Q   # Fixed beam width throughout
    num_return_candidates = args.N if args.N is not None else beam_width  # Number of final candidates to return
    final_candidates = []  # Completed sequences
    tokens_generated = 0

    assert type(value_generator) is ValueGeneratorBatched, "ValueGeneratorBatched must be used for beam search"
    gen_past_kvs = None

    value_generator.reset() # Empty the value models' caches from previous prompts
    
    with torch.no_grad():
        while tokens_generated < max_completion_len:# and len(beams) > 0:
            all_continuations = []  # All candidate continuations
            if tokens_generated == 0: # Equivalent to "if gen_past_kvs is None:"
                model_outputs = generative_model(input_ids=input_ids, use_cache=True, return_dict=True)
            else:
                # By this point, we will have a full set of candidates
                # For Llama models, it is required to give cache_position as an argument if you are passing in a cache (via past_key_values)
                # cache_position should be the index for the new token which was not present in the last forward pass; this is simply 
                # y.shape[1]-1 for the unbatched case, but we need to provide it as a tensor
                model_inputs = generative_model.prepare_inputs_for_generation(input_ids, past_key_values=gen_past_kvs, cache_position=torch.tensor(input_ids.shape[1]-1).unsqueeze(0).to(args.device), use_cache=True)
                model_outputs = generative_model(**model_inputs, return_dict=True)
            logits = model_outputs.logits
            gen_past_kvs = model_outputs.past_key_values

            # Get next token probabilities
            model_outputs = generative_model(input_ids=input_ids, use_cache=True, return_dict=True)
            logits = model_outputs.logits

            top_k_probs, top_k_indices = top_k_from_logits(logits, args.topk, args.temperature)
            
            # # Use this to replicate the original MAVIS behavior
            # for i in range(top_k_indices.shape[0]):
            #     if i >= 1:
            #         top_k_indices[i, :] = tokenizer.pad_token_id
            
            values = value_generator.get_values(input_ids, top_k_indices).to(g_device)
            if tokens_generated == 0:
                # Beam dimension is not automatically added on the first iteration, but we need it for consistency
                values = values.unsqueeze(0)
            values_unnormalized = values.clone()  # Save unnormalized values for ranking candidates across beams
            for i in range(input_ids.shape[0]):
                values[i, top_k_indices[i] != tokenizer.pad_token_id, 0] = normalize_values(values[i, top_k_indices[i] != tokenizer.pad_token_id, :], 1).squeeze()
                values[i, top_k_indices[i] == tokenizer.pad_token_id, 0] = -torch.inf
                values_unnormalized[i, top_k_indices[i] == tokenizer.pad_token_id, 0] = -torch.inf
                # break # Uncomment to reproduce the original MAVIS behavior with a single beam

            exp_values = torch.exp(args.beta * values).squeeze()
            # Reshape exp_values to match top_k_probs shape
            exp_values = exp_values.reshape(top_k_probs.shape[0], top_k_probs.shape[1])

            # Multiply reference probabilities with exp_values
            # This is the π₀*e^(βv) formula
            modified_probs = top_k_probs * exp_values

            # Normalize to make it a proper probability distribution
            if modified_probs.sum() <= 1e-6:
                # Avoid division by too small value
                modified_probs /= (modified_probs.sum(1, keepdim=True) + 1e-6)
            else:
                modified_probs /= modified_probs.sum(1, keepdim=True)

            # CHANGE 3: Sample 2Q tokens with replacement
            # Setting num_to_sample to 2 works much better than beam_width * 2 currently
            # We set replacement to true so that if there is only one option for one beam but multiple options for another,
                # it wont try to sample the padding indices
                # TODO: increase num_to_sample when more beams are added
            num_to_sample = min(1, top_k_indices.shape[1])
            # num_to_sample = 1 # Uncomment to reproduce the original MAVIS behavior with a single beam
            sampled_indices = torch.multinomial(modified_probs, num_samples=num_to_sample, replacement=True)
            sampled_indices = sampled_indices

            # CHANGE 4: Use combined score (value*beta + log_prob)
            # TODO: Confirm that this is better than just ranking using the value (in particular, the moving window
            # average of unnormalized values)
            # I think that taking the log of the probability is incorrect here because we will be making comparisons between
            # beams, and the probabilities came from softmaxes with different denominators for each beam. It would be better
            # to use the logits directly from the base model, or the overall probability of each sequence
            # Currently, the score is just the unnormalized value
            combined_scores = values_unnormalized[torch.arange(values_unnormalized.shape[0]).unsqueeze(1), sampled_indices].squeeze(2)
            is_eos = (top_k_indices == eos_token_id)
            for i in range(combined_scores.shape[0]):
                for j in range(combined_scores.shape[1]):
                    sampled_index = sampled_indices[i,j]
                    if top_k_indices[i,sampled_index] == tokenizer.pad_token_id:
                        # If the combined score is -inf, it means that the token is a padding token
                        # We can skip this token
                        continue
                    seq = torch.cat([input_ids[i], top_k_indices[i,sampled_index].unsqueeze(0)], dim=0)
                    val = combined_scores[i,j].item()  # Use unnormalized value for selecting the final sequences to send to the reward model
                    all_continuations.append((seq, val, is_eos[i,sampled_index], combined_scores[i,j], i))
        
            # No valid continuations found
            if not all_continuations:
                break

            # Sort by combined score (highest first)
            all_continuations.sort(key=lambda x: x[3], reverse=True)
            
            # Separate completed and continuing sequences
            completed_sequences = []
            continuing_sequences = []
            
            for seq, val, is_eos, score, beam_idx in all_continuations:
                if is_eos:
                    completed_sequences.append((seq, val, score))
                else:
                    continuing_sequences.append((seq, val, score, beam_idx))
            
            final_candidates.extend(completed_sequences)

            if len(continuing_sequences) == 0:
                assert len(final_candidates) > 0, "No valid continuations found and no completed sequences"
                break
            elif len(continuing_sequences) < beam_width:
                # If we have fewer continuing sequences than the desired beam width, duplicate the best ones
                    # Note that continuing_sequences is already sorted by score
                i = 0
                while len(continuing_sequences) < beam_width:
                    # Duplicate the best sequence
                    continuing_sequences.append(continuing_sequences[i])
                    i += 1

            # CHANGE 5: Always select top beam_width continuing sequences
            # (don't reduce beam width when sequences complete)
            new_beams = []
            new_beam_indices = []
            # new_beam_probs = []
            for seq, val, _, beam_idx in continuing_sequences[:beam_width]:
                new_beams.append(seq)
                new_beam_indices.append(beam_idx)
                new_token = seq[-1].item()
                # new_beam_probs.append(top_k_probs[beam_idx][top_k_indices[beam_idx] == new_token].item())

            # Need to reorder the KV cache based on the new beams
            beams = new_beams
            # No more sequences to continue
            if not beams:
                break
            else:
                # Update input_ids for the next iteration
                input_ids = torch.stack(beams, dim=0)
                if tokens_generated == 0:
                    # Expand the KV cache to accommodate the desired number of beams
                    gen_past_kvs.batch_repeat_interleave(input_ids.shape[0])
                    value_generator.expand_beams(input_ids.shape[0])
                else:
                    # Update the KV cache based on the new beam indices
                    # Note that this does the same thing as gen_past_kvs.reorder_cache(torch.tensor(new_beam_indices))
                    old_gen_kvs_split = gen_past_kvs.batch_split(len(new_beam_indices), 1)
                    new_gen_kvs_split = [old_gen_kvs_split[i] for i in new_beam_indices]
                    gen_past_kvs.from_batch_splits(new_gen_kvs_split)
                    value_generator.update_beams(new_beam_indices)

            tokens_generated += 1
            if tokens_generated % 20 == 0:
                print(f"Tokens: {tokens_generated}, Beam size: {len(beams)}, Completed: {len(final_candidates)}")
            
        # Add any remaining beam sequences to final candidates
        final_candidates.extend([(seq, val, score) for seq, val, score, beam_idx in continuing_sequences])
    
    # Sort final candidates by score for best selection
    if len(final_candidates) > 1:
        sorted_final_candidates = sorted(final_candidates, key=lambda x: x[2], reverse=True)
        final_texts = [tokenizer.decode(candidate[0], skip_special_tokens=True) for candidate in sorted_final_candidates[:num_return_candidates]]
        return final_texts
    else:
        return [tokenizer.decode(final_candidates[0][0], skip_special_tokens=True)]

log_dir = os.path.join(args.log_dir, "_".join(objectives), datetime.now().strftime("%m-%d_%H-%M") + f"_{args.Q}beams/")
file_name = os.path.join(log_dir, f"mavis_{args.obj_weights}.csv")
os.makedirs(log_dir, exist_ok=True)

# Save the arguments to a YAML file
with open(os.path.join(log_dir, f"mavis_bs_{args.obj_weights}_args.yaml"), "w") as f:
    yaml.dump(vars(args), f)

if __name__ == "__main__":
    eval_loop(inputs, prompt_indices, args.num_trials, mavis_beam, reward_model, rm_device, get_rewards, obj_weights=obj_weights, 
              verbose=True, track_time=args.track_time, file_name=file_name,)
    print("**********************************")
    print(f"Results saved to {file_name}")
    print("**********************************")