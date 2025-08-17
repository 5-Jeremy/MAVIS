import numpy as np
import torch
import os, yaml
from peft import PeftModel
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
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
A.add_argument("--track_time", action="store_true")
A.add_argument("--num_samples", type=int, default=1) 
A.add_argument("--ppo", action="store_true")
A.add_argument("--morlhf", action="store_true", help="Use the MORLHF model instead of the reward soup mix")
A.add_argument("--seed", type=int, default=0, help="The random seed is set to this at the start of each prompt")
A.add_argument("--log_dir", type=str, default="baseline_logs/" + datetime.now().strftime("%m-%d_%H-%M"))
args = A.parse_args()

verbose = True
prompt_indices = list(range(args.num_prompts))

print("Dataset:", args.dataset)
print(f"Prompt indices: {prompt_indices[0]} to {prompt_indices[-1]}")
print("Number of samples (N):", args.num_samples)

g_device, rm_device = assign_gpus()

if args.dataset == "anthropic":
    dataset_valid_objectives = ["help", "harm", "humor"]
elif args.dataset == "summary":
    dataset_valid_objectives = ["summarization", "faithful"]
else:
    raise ValueError("Invalid dataset_type")

# Parse the given objective weights
obj_weights = parse_objective_weights(args.obj_weights, dataset_valid_objectives)

objectives = [obj for obj, weight in obj_weights.items() if weight is not None]

# Reward and value models start in the CPU to avoid OOM
if args.dataset == "anthropic":
    loaded_assets = load_for_anthropic(
        csv_path="datasets/anthropic/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights, ppo=args.ppo,
        morlhf=args.morlhf,
    )
elif args.dataset == "summary":
    loaded_assets = load_for_summary(
        csv_path="datasets/summary/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights, ppo=args.ppo,
        morlhf=args.morlhf,
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
        f"sft_model/{args.dataset}/",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    return model.to(g_device)

ppo_paths = [f"morlhf/{args.dataset}/single/{obj}" for obj in objectives]
ppo_models = [load_model(path) for path in ppo_paths]
weights = [weight for obj, weight in obj_weights.items() if weight is not None]

# NOTE: Only the models given to the logits processor will actually influence what is generated.
class ModelMixtureLogitsProcessorNoCache(LogitsProcessor):
    """LogitsProcessor that mixes logits from ppo_models models without KV caching"""
    def __init__(self, ppo_models,temperature=1.0, debug=False):
        self.ppo_models = ppo_models
        self.temperature = temperature
        
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
        return mixed_logits  

def mod(prompt_str):
    """
    Mix the ppo logits according to the weights.
    """
    prompt = tokenizer(prompt_str, return_tensors="pt").to(g_device)


    logits_processor = ModelMixtureLogitsProcessorNoCache(
        ppo_models=ppo_models,
        temperature=args.temperature
    )
    
    # Generate N samples
    with torch.no_grad():
        outputs = generative_model.generate(
            prompt["input_ids"],
            do_sample=True,
            top_k=args.topk,
            temperature=args.temperature,
            max_new_tokens=max_completion_len,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=args.num_samples,
            logits_processor=LogitsProcessorList([logits_processor])
        )
    
    # Decode the generated sequences
    candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return candidates

log_dir = os.path.join(args.log_dir, "_".join(objectives), datetime.now().strftime("%m-%d_%H-%M"))

log_dir = args.log_dir
if args.ppo:
    file_name = os.path.join(log_dir, f"results_ppo_{args.obj_weights}.csv")
elif args.morlhf:
    file_name = os.path.join(log_dir, f"results_morlhf_{args.obj_weights}.csv")
else:
    file_name = os.path.join(log_dir, "results_base.csv")
os.makedirs(log_dir, exist_ok=True)

if __name__ == "__main__":
    eval_loop(inputs, prompt_indices, args.num_trials, mod, reward_model, rm_device, get_rewards, obj_weights=obj_weights,
              obj_weights_str=args.obj_weights, verbose=True, track_time=args.track_time, file_name=file_name, seed=args.seed)