from __future__ import annotations

import argparse
import json, yaml
import sys, os
from datetime import datetime
from utils.LLM_judge_utils import get_judge_prompt_helpful_harmless, get_judgements, load_json

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two JSON files by top-level keys and extract 'completion' from subkey '1'."
    )
    parser.add_argument("file_a", help="Path to first JSON file")
    parser.add_argument("file_b", help="Path to second JSON file")
    parser.add_argument("--comparison_name", type=str, default=datetime.now().strftime("%m-%d_%H-%M"), help="Directory to log raw outputs from the judge LLM as well as the final win rate.")
    parser.add_argument("--dataset", type=str, default="anthropic", choices=["anthropic", "safeRLHF"], help="Dataset type to determine comparison method.")
    parser.add_argument("--single_only", action="store_true", help="If set, evaluate only on a single dimension (helpfulness or harmlessness) specified using --objective.")
    parser.add_argument("--objective", type=str, default=None, choices=["helpfulness", "harmlessness"], help="Preference dimension on which to judge the completions.")
    args = parser.parse_args()

    if not args.single_only:
        assert args.objective is None, "Do not provide --objective unlesss --single_only is used."
        args.objective = "mix"

    os.makedirs(os.path.join("Eval_Logs/LLM_judge_logs", args.comparison_name, args.objective, "raw_comparisons"), exist_ok=True)
    os.makedirs(os.path.join("Eval_Logs/LLM_judge_logs", args.comparison_name, args.objective, "raw_comparisons_reverse"), exist_ok=True)

    a = load_json(args.file_a)
    b = load_json(args.file_b)

    if not isinstance(a, dict) or not isinstance(b, dict):
        print("Both JSON files must contain a top-level object/dictionary.", file=sys.stderr)
        sys.exit(2)

    keys_a = set(a.keys())
    keys_b = set(b.keys())
    if keys_a != keys_b:
        missing_in_b = sorted(list(keys_a - keys_b))
        missing_in_a = sorted(list(keys_b - keys_a))
        msg = (
            "Top-level keys differ between the two files."
            f" missing_in_b={missing_in_b}, missing_in_a={missing_in_a}"
        )
        raise AssertionError(msg)

    # Save the arguments to a YAML file
    with open(os.path.join("Eval_Logs/LLM_judge_logs", args.comparison_name, args.objective, f"args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Get the set of prompts for the judge LLM
    all_judge_prompts = {}
    all_judge_prompts_reverse = {}
    for key in sorted(keys_a):
        val_a = a[key]
        val_b = b[key]

        if not isinstance(val_a, dict) or not isinstance(val_b, dict):
            raise AssertionError(
                f"Value for top-level key {key!r} must be a dict in both files"
            )

        if "0" not in val_a:
            raise AssertionError(f"In {args.file_a}, top-level key {key!r}: missing subkey '0'")
        if "0" not in val_b:
            raise AssertionError(f"In {args.file_b}, top-level key {key!r}: missing subkey '0'")

        entry_a = val_a["0"]
        entry_b = val_b["0"]

        if not isinstance(entry_a, dict) or not isinstance(entry_b, dict):
            raise AssertionError(
                f"Entry '0' under top-level key {key!r} must be a dict in both files"
            )

        comp_a = entry_a.get("completion")
        comp_b = entry_b.get("completion")

        # Separate prompt from completions
        if args.dataset == "anthropic":
            assert args.objective in ["helpfulness", "harmlessness", "mix"], "For Anthropic dataset, must choose to evaluate on a mix of helpfulness and harmlessnes (the default), or just one specific objective."
            assert "Human: " in comp_a and "Assistant:" in comp_a
            assert "Human: " in comp_b and "Assistant:" in comp_b
            prompt = comp_a.split("Human: ")[1].split("Assistant:")[0].strip()
            comp_a = comp_a.split("Assistant:")[1].strip()
            comp_b = comp_b.split("Assistant:")[1].strip()
            full_prompt = get_judge_prompt_helpful_harmless(prompt, comp_a, comp_b, objective=args.objective)
            full_prompt_reverse = get_judge_prompt_helpful_harmless(prompt, comp_b, comp_a, objective=args.objective)
        elif args.dataset == "safeRLHF":
            assert args.objective in ["helpfulness", "harmlessness", "mix"], "For SafeRLHF dataset, must choose to evaluate on a mix of helpfulness and harmlessnes (the default), or just one specific objective."
            assert "BEGINNING OF CONVERSATION: USER: " in comp_a and "ASSISTANT:" in comp_a
            assert "BEGINNING OF CONVERSATION: USER: " in comp_b and "ASSISTANT:" in comp_b
            prompt = comp_a.split("BEGINNING OF CONVERSATION: USER: ")[1].split("ASSISTANT:")[0].strip()
            comp_a = comp_a.split("ASSISTANT:")[1].strip()
            comp_b = comp_b.split("ASSISTANT:")[1].strip()
            full_prompt = get_judge_prompt_helpful_harmless(prompt, comp_a, comp_b, objective=args.objective)
            full_prompt_reverse = get_judge_prompt_helpful_harmless(prompt, comp_b, comp_a, objective=args.objective)
        else:
            raise ValueError("Invalid dataset type")
        all_judge_prompts[key] = full_prompt
        all_judge_prompts_reverse[key] = full_prompt_reverse

    # Now query the judge LLM on the vllm inference server for each prompt and tally results
    a_wins = 0
    b_wins = 0
    judgements = get_judgements(all_judge_prompts, log_dir=os.path.join("Eval_Logs/LLM_judge_logs", args.comparison_name, args.objective, "raw_comparisons"))
    judgements_reverse = get_judgements(all_judge_prompts_reverse, log_dir=os.path.join("Eval_Logs/LLM_judge_logs", args.comparison_name, args.objective, "raw_comparisons_reverse"))
    num_ties = 0
    for key, judgement in judgements.items():
        if judgement is None:
            print(f"Could not parse judge response for key {key!r}", file=sys.stderr)
        elif judgement == '1':
            a_wins += 1
        elif judgement == '2':
            b_wins += 1
        elif judgement == 'tie':
            num_ties += 1
    for key, judgement in judgements_reverse.items():
        if judgement is None:
            print(f"Could not parse judge response for key {key!r} (reverse)", file=sys.stderr)
        elif judgement == '1':
            b_wins += 1
        elif judgement == '2':
            a_wins += 1
        elif judgement == 'tie':
            num_ties += 1
    print("Number of ties:", num_ties)
    total = a_wins + b_wins + num_ties
    assert total > 0, "No wins recorded for a or b."
    print(f"Final results over {total} comparisons:")
    print(f"File A win rate: {a_wins}/{total} = {a_wins/total:.2%}")
    print(f"File B win rate: {b_wins}/{total} = {b_wins/total:.2%}")
    results_file = os.path.join("Eval_Logs/LLM_judge_logs", args.comparison_name, args.objective, "final_results.txt")
    with open(results_file, "w", encoding="utf-8") as rf:
        rf.write(f"Final results over {total} comparisons:\n")
        rf.write(f"File A win rate: {a_wins}/{total} = {a_wins/total:.2%}\n")
        rf.write(f"File B win rate: {b_wins}/{total} = {b_wins/total:.2%}\n")


if __name__ == "__main__":
    main()
