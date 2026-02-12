#!/usr/bin/env python3
"""
Script to merge multiple model weights with specified preference coefficients.
Uses the merge_weights_with_preference function to combine models.
"""

import argparse
import os
import sys
import torch
import gc
import shutil
from transformers import AutoModelForCausalLM
from peft import PeftModel

import psutil
import os

# Could be useful for checking if you are about to run out of memory
def get_current_memory_usage_mb():
    """
    Returns the resident set size (RSS) memory usage of the current process in MB.
    """
    process = psutil.Process(os.getpid())
    # memory_info() provides various memory statistics; rss is resident set size
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

def get_average_state_dict(state_dicts, coefficients):
    """Average state dicts with given coefficients."""
    i = 0
    averaged_state_dict = {}
    for state_dict, coefficient in zip(state_dicts, coefficients):
        current_weights = state_dict
        for key in list(current_weights.keys()):
            if i == 0:
                averaged_state_dict[key] = coefficient * current_weights[key]
            else :
                averaged_state_dict[key] += coefficient * current_weights[key]
        i += 1
    return averaged_state_dict

def merge_weights_with_preference(base_model_names, preference, temp_save_path, sft_model_path):
    """
    Merge multiple model weights with given preference coefficients.
    
    Args:
        base_model_names: List of paths to PEFT adapter models
        preference: List of coefficients for weighted averaging
        temp_save_path: Path to save the merged model
        sft_model_path: Path to the base SFT model
    """
    models = []
    for base_model_name in base_model_names:
        print(f"Loading SFT model from: {sft_model_path}")
        print(f"Applying PEFT adapter from: {base_model_name}")
        
        # Load the base SFT model
        model_tmp = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            device_map='cpu',
        )
        
        # Apply the PEFT adapter
        model_tmp = PeftModel.from_pretrained(model_tmp, base_model_name)
        
        # Merge and unload the adapter
        model_tmp = model_tmp.merge_and_unload()
        
        models.append(model_tmp)
    
    print("Extracting state dictionaries...")
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    
    print(f"Merging with preferences: {preference}")
    average_weights = get_average_state_dict(state_dicts, preference)
    
    print("Loading merged weights into first model...")
    model_1 = models[0]
    model_1.load_state_dict(average_weights, strict=False)
    
    if os.path.exists(temp_save_path):
        print(f"Removing existing directory: {temp_save_path}")
        shutil.rmtree(temp_save_path, ignore_errors=True)
    
    print(f"Saving merged model to: {temp_save_path}")
    model_1.save_pretrained(temp_save_path)

    # Clean up memory
    print("Cleaning up memory...")
    while len(models):
        del models[0]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()
    print("Model merging completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple model weights with preference coefficients",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument(
        "--sft_model", 
        type=str, 
        required=True,
        help="Path to the base SFT model"
    )
    parser.add_argument(
        "--model1", 
        type=str, 
        required=True,
        help="Path to the first PEFT adapter weights"
    )
    parser.add_argument(
        "--model2", 
        type=str, 
        required=True,
        help="Path to the second PEFT adapter weights"
    )
    parser.add_argument(
        "--model3", 
        type=str, 
        required=False,
        help="Path to the third PEFT adapter weights"
    )
    
    # Preference weights
    parser.add_argument(
        "--pref1", 
        type=float, 
        required=True,
        help="Preference weight for the first model"
    )
    parser.add_argument(
        "--pref2", 
        type=float, 
        required=True,
        help="Preference weight for the second model"
    )
    parser.add_argument(
        "--pref3", 
        type=float, 
        required=False,
        help="Preference weight for the third model"
    )
    
    # Output path
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Path to save the merged model"
    )
    
    # Optional: Validate preference weights sum to 1
    parser.add_argument(
        "--normalize", 
        action="store_true",
        help="Normalize preference weights to sum to 1.0"
    )
    
    args = parser.parse_args()
    
    # Validate that SFT model path exists
    if not os.path.exists(args.sft_model):
        print(f"Error: SFT model path does not exist: {args.sft_model}")
        sys.exit(1)
    
    # Validate that all PEFT adapter paths exist
    if args.model3 is None:
        model_paths = [args.model1, args.model2]
        preference_weights = [args.pref1, args.pref2]
    else:
        model_paths = [args.model1, args.model2, args.model3]
        preference_weights = [args.pref1, args.pref2, args.pref3]
    for i, path in enumerate(model_paths, 1):
        if not os.path.exists(path):
            print(f"Error: PEFT adapter {i} path does not exist: {path}")
            sys.exit(1)
    
    # Normalize if requested
    if args.normalize:
        total = sum(preference_weights)
        if total == 0:
            print("Error: Sum of preference weights cannot be zero when normalizing")
            sys.exit(1)
        preference_weights = [w / total for w in preference_weights]
        print(f"Normalized preference weights: {preference_weights}")
    else:
        # Check if weights sum to approximately 1.0 (warn if not)
        total = sum(preference_weights)
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Preference weights sum to {total:.3f}, not 1.0")
            print("Consider using --normalize flag if you want them normalized")
    
    print(f"SFT model path: {args.sft_model}")
    print(f"PEFT adapter paths: {model_paths}")
    print(f"Preference weights: {preference_weights}")
    print(f"Output path: {args.output}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Perform the merge
    merge_weights_with_preference(model_paths, preference_weights, args.output, args.sft_model)


if __name__ == "__main__":
    main()
