import numpy as np
import torch
import os, yaml
from peft import PeftModel
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from time import time
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)

from utils.loading_utils import load_for_anthropic, load_for_summary
from utils.search_utils import assign_gpus, eval_loop, parse_objective_weights
from utils.gen_utils import add_generation_args

A = argparse.ArgumentParser()
add_generation_args(A, {"data_split":"test"}) # dataset, data_split, temperature, topk, device, no_cache
A.add_argument("--obj_weights", type=str, default="1.0,0.0,0.0", help="Comma-separated list of weights for the three objectives (Help, Harm, Humor for the anthropic dataset). If you do not want to evaluate on an objective, leave it empty (e.g. 0.5,,0.5).")
A.add_argument("--num_prompts", type=int, default=100, help="Number of prompts to evaluate on; the prompts are selected in order starting from index 0")
A.add_argument("--num_trials", type=int, default=3)
A.add_argument("--num_samples", type=int, default=1) 
A.add_argument("--seed", type=int, default=0, help="The random seed is set to this at the start of each prompt")
A.add_argument("--log_dir", type=str, default="Eval_Logs/baseline_logs/" + datetime.now().strftime("%m-%d_%H-%M"))
A.add_argument("--alt_device", type=str, default="cuda:1", help="Device for the models used by the logits processor (the PPO models)")
args = A.parse_args()

verbose = True
prompt_indices = list(range(args.num_prompts))

print("Dataset:", args.dataset)
print(f"Prompt indices: {prompt_indices[0]} to {prompt_indices[-1]}")
print("Number of samples (N):", args.num_samples)

g_device, rm_device = assign_gpus(devices=[args.device, args.alt_device])

if args.dataset == "anthropic":
    dataset_valid_objectives = ["help", "harm", "humor"]
elif args.dataset == "summary":
    dataset_valid_objectives = ["summarization", "faithful"]
else:
    raise ValueError("Invalid dataset_type")

# Parse the given objective weights
obj_weights = parse_objective_weights(args.obj_weights, dataset_valid_objectives)

objectives = [obj for obj, weight in obj_weights.items() if weight is not None]

# NOTE: Here we only load the SFT model. The PPO models are loaded later and come in through the ModelMixtureLogitsProcessorNoCache
if args.dataset == "anthropic":
    loaded_assets = load_for_anthropic(
        csv_path="datasets/anthropic/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights, ppo=False,
        morlhf=False, base_model_type=args.base_model_type,
    )
elif args.dataset == "summary":
    loaded_assets = load_for_summary(
        csv_path="datasets/summary/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights, ppo=False,
        morlhf=False, base_model_type=args.base_model_type,
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

def load_model(path):
    base = AutoModelForCausalLM.from_pretrained(
        f"Models/sft_model/{args.base_model_type}/{args.dataset}/",
        torch_dtype=torch.bfloat16,
        device_map=args.alt_device,
    )
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    return model.to(g_device)

ppo_paths = [f"Models/morlhf/{args.base_model_type}/{args.dataset}/single/{obj}" for obj in objectives]
ppo_models = [load_model(path) for path in ppo_paths]
weights = [weight for obj, weight in obj_weights.items() if weight is not None]

# NOTE: Only the models given to the logits processor will actually influence what is generated.
class ModelMixtureLogitsProcessorNoCache(LogitsProcessor):
    """LogitsProcessor that mixes logits from ppo_models models without KV caching"""
    def __init__(self, ppo_models,temperature=1.0):
        self.ppo_models = ppo_models
        self.temperature = temperature
        self.ref_logits = []
        
    def __call__(self, input_ids, scores):
        """Mix logits from both models according to mix_ratio"""
        with torch.no_grad():
            # Process full input each time without caching
            self.ppo_outputs = [model(input_ids=input_ids, use_cache=False) for model in self.ppo_models]
        # Get last token logits from both models
        self.ppo_logits = [output.logits[:, -1, :] for output in self.ppo_outputs]
        
        # Mix logits according to the ratio
        # use the weights and the ppo_logits to get the mixed_logits
        mixed_logits = torch.zeros_like(self.ppo_logits[0])
        for i, logits in enumerate(self.ppo_logits):
            mixed_logits += weights[i] * logits

        # scores is just the logits from the reference model
        # TODO: I'm not sure this is compatible with batched inference
        self.ref_logits.append(scores)
        return mixed_logits

global sequence_kl_div_list
sequence_kl_div_list = []

def mod(prompt_str):
    """
    Mix the ppo logits according to the weights.
    """
    global sequence_kl_div_list
    prompt = tokenizer(prompt_str, return_tensors="pt").to(g_device)
    num_prompt_tokens = prompt["input_ids"].shape[1]
    return_stats = {}

    logits_processor = ModelMixtureLogitsProcessorNoCache(
        ppo_models=ppo_models,
        temperature=args.temperature
    )
    
    # Generate N samples
    with torch.no_grad():
        generation_start_time = time()
        outputs = generative_model.generate(
            prompt["input_ids"],
            do_sample=True,
            top_k=args.topk,
            temperature=args.temperature,
            max_new_tokens=max_completion_len,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=args.num_samples,
            logits_processor=LogitsProcessorList([logits_processor]),
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
    generation_time_elapsed = time() - generation_start_time
    # All tokens per second statistics reported in the paper assume a batch size of 1; we do not adjust for args.num_samples here
    return_stats["tokens_per_second"] = (outputs.sequences.shape[1] - num_prompt_tokens) / generation_time_elapsed
    chosen_tokens = outputs.sequences[0, num_prompt_tokens:]

    baseline_scores = outputs.scores
    baseline_scores_tensor = torch.stack(baseline_scores, dim=1)  # Shape: (batch_size, num_new_tokens, vocab_size)
    baseline_score_probs = torch.softmax(baseline_scores_tensor, dim=-1)
    
    assert args.num_samples == 1, "Currently only supports num_samples=1 to properly compute ref probs"
    ref_logits = torch.concatenate(logits_processor.ref_logits, dim=0)
    ref_probs = torch.softmax(ref_logits/args.temperature, dim=-1).unsqueeze(0)  # Shape: (1, num_new_tokens, vocab_size)
    
    # We compute the log probability ratio for each token and sum them up to get the sequence log probability ratio
    sequence_logprob_ratio = 0
    for i in range(len(baseline_scores)):
        token_prob_ratio = baseline_score_probs[:,i, chosen_tokens[i].item()] / ref_probs[:,i, chosen_tokens[i].item()]
        sequence_logprob_ratio += torch.log(token_prob_ratio)

    sequence_kl_div_list.append(sequence_logprob_ratio.item())
    print(f"Running average KL divergence: {np.mean(sequence_kl_div_list)}")
    return_stats["sequence_KL"] = sequence_logprob_ratio.item()

    # Decode the generated sequences
    candidates = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    
    return candidates, return_stats

log_dir = os.path.join(args.log_dir, "_".join(objectives), datetime.now().strftime("%m-%d_%H-%M"))

log_dir = args.log_dir
file_name = os.path.join(log_dir, f"results_mod_{args.obj_weights}.csv")
os.makedirs(log_dir, exist_ok=True)

# Save the arguments to a YAML file
with open(os.path.join(log_dir, f"args.yaml"), "w") as f:
    yaml.dump(vars(args), f)

if __name__ == "__main__":
    if args.num_samples > 1:
        print("WARNING: Tokens per second statistics are calculated assuming a batch size of 1; since num_samples > 1, the statistics will not be accurate.")
    eval_loop(inputs, prompt_indices, args.num_trials, mod, reward_model, rm_device, get_rewards, obj_weights=obj_weights,
              verbose=True, track_KL=True, file_name=file_name, seed=args.seed, save_json=(args.num_trials==1 and args.num_samples==1))
    
# Does the logit processor get applied before or after the scores are generated?
# In other words, do we need to compute baseline_score_probs any differently here compared to in baseline_eval_kl.py?