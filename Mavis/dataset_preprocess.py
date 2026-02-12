import os
import pandas as pd
from datasets import load_dataset
import argparse
from transformers import AutoTokenizer
import json
from utils.loading_utils import make_full_prompt_summary, load_sft_tokenizer, format_prompt_pku_safe_rlhf

A = argparse.ArgumentParser()
A.add_argument("dataset", type=str, choices=["anthropic", "summary", "safeRLHF"])
A.add_argument("split", type=str, choices=["train", "test"])
args = A.parse_args()

if args.dataset in ["anthropic", "summary"]:
    tokenizer_style = "llama"
elif args.dataset == "safeRLHF":
    tokenizer_style = "alpaca"

gen_tokenizer = load_sft_tokenizer(tokenizer_style=tokenizer_style)

if args.dataset == "anthropic":
    dataset = load_dataset("Anthropic/hh-rlhf", split=args.split)
    dataset = dataset.map(lambda x: {"prompt": x["chosen"][:x["chosen"].find("Assistant: ")+len("Assistant: ")]})
    # Filter out prompts that are longer than 200 tokens
    dataset = dataset.filter(lambda x: len(gen_tokenizer(x["prompt"])["input_ids"]) <= 200)
elif args.dataset == "summary":
    # We use the validation split for this dataset as a test split
    if args.split == "test":
        split = "validation"
    else:
        split = args.split
    dataset = load_dataset("openai/summarize_from_feedback", 'comparisons')[split]
    # Same method as used in the MOD paper
    dataset = dataset.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200)
    dataset = dataset.map(make_full_prompt_summary)
    # Filter out prompts that are longer than 512 tokens or shorter than 8 tokens
    dataset = dataset.filter(lambda x: len(gen_tokenizer(x["prompt"])["input_ids"]) <= 512)
    dataset = dataset.filter(lambda x: len(gen_tokenizer(x["prompt"])["input_ids"]) >= 8)
elif args.dataset == "safeRLHF":
    if args.split == "train":
        # We use prompts from the same preference dataset that PARM uses. They trained on the first 8000 prompts and
        # tested on prompts starting at index 8500 (all coming from the train split of the PKU-SafeRLHF-10K dataset)
        # We do not need to filter by length because we are only using the prompts, not the example completions
        dataset = load_dataset(
            "PKU-Alignment/PKU-SafeRLHF-10K",
            split="train",
            num_proc=2,
        )
        original_columns = dataset.column_names
        # Add the correct template
        dataset = dataset.map(lambda x: {"prompt": format_prompt_pku_safe_rlhf(x["prompt"], eos_token='Not used')}, remove_columns=original_columns)
        dataset = dataset[:8000]
    else:
        dataset = json.load(open("datasets/safeRLHF/test_prompt_only.json", "r"))
        dataset = [{"prompt": format_prompt_pku_safe_rlhf(item["prompt"], eos_token='Not used')} for item in dataset]

dataset_pd = pd.DataFrame(dataset)
# There is a duplicate in the first 100 prompts that PARM evaluates on, so I avoid dropping duplicates in the data that MAVIS is evaluated on to ensure consistency
if not (args.dataset == "safeRLHF" and args.split == "test"):
    dataset_pd.drop_duplicates(subset=["prompt"], inplace=True)
dataset_pd = dataset_pd.loc[:, ["prompt"]]
output_dir = f"datasets/{args.dataset}/"
os.makedirs(output_dir, exist_ok=True)
dataset_pd.to_csv(os.path.join(output_dir, f"{args.dataset}_{args.split}_deduped.csv"), index=False)