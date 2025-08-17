import os
import pandas as pd
from datasets import load_dataset
import argparse
from transformers import AutoTokenizer
from utils.loading_utils import make_full_prompt_summary

A = argparse.ArgumentParser()
A.add_argument("dataset", type=str)
A.add_argument("split", type=str, choices=["train", "test"])
args = A.parse_args()

if args.dataset == "anthropic":
    gen_tokenizer = AutoTokenizer.from_pretrained("sft_model/anthropic")
    dataset = load_dataset("Anthropic/hh-rlhf", split=args.split)
    dataset = dataset.map(lambda x: {"prompt": x["chosen"][:x["chosen"].find("Assistant: ")+len("Assistant: ")]})
    # Filter out prompts that are longer than 200 tokens
    dataset = dataset.filter(lambda x: len(gen_tokenizer(x["prompt"])["input_ids"]) <= 200)
elif args.dataset == "summary":
    gen_tokenizer = AutoTokenizer.from_pretrained("sft_model/anthropic")
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

dataset_pd = pd.DataFrame(dataset)
dataset_pd.drop_duplicates(subset=["prompt"], inplace=True)
dataset_pd = dataset_pd.loc[:, ["prompt"]]
output_dir = f"datasets/{args.dataset}/"
os.makedirs(output_dir, exist_ok=True)
dataset_pd.to_csv(os.path.join(output_dir, f"{args.dataset}_{args.split}_deduped.csv"), index=False)