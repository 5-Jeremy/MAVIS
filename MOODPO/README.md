## MOODPO (Multi-Objective Online DPO)

This directory contains a MOODPO training script built on top of TRL’s **Online DPO** trainer.

### Overview

MOODPO trains a policy by sampling a weight vector over multiple reward models (e.g. helpful/harmless/humor), injecting
those weights into the prompt as a system instruction, generating two completions per prompt, scoring them with the
weighted rewards, and optimizing an Online DPO objective.

The training entrypoint is `moodpo.py`.

### Prerequisites

- **Python**: 3.10+
- **GPU**: recommended (training and reward models are typically GPU-bound)

### Installation (from repo root)

Follow the TRL repo setup instructions first, then install dependencies:

```bash
pip install -U pip
pip install -e ".[dev]"
pip install wandb
```

Configure Accelerate once (recommended):

```bash
accelerate config
```

### Dataset format

`MOODPO/moodpo/moodpo.py` expects a **CSV** containing prompts. By default it reads the `prompt` column.

Example:

```csv
prompt
Tell me a joke about penguins.
Explain backpropagation simply.
```

If your column name differs, pass `--prompt_column <COLUMN_NAME>`.

### Running

Run commands from the **repo root** so local imports resolve correctly.

#### Important note for Iteration 1

`MOODPO/moodpo/moodpo.py` currently sets a machine-specific default value for `--model_path`. For a true “fresh start”
run, ensure `--model_path` is **not set** (or update the default in the script to `None`) so it doesn’t attempt to load
an adapter checkpoint.

#### Iteration 1 (fresh start)

```bash
accelerate launch MOODPO/moodpo/moodpo.py \
  --wandb_name "moodpo_iter1" \
  --sft_model_path "<BASE_OR_SFT_MODEL_ID_OR_PATH>" \
  --csv_path "/absolute/path/to/prompts_train.csv" \
  --prompt_column "prompt" \
  --reward_models helpful harmless humor \
  --output_dir "MOODPO/outputs/iter1" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-7 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True
```

#### Iteration 2 (resume from Iteration 1)

Select an Iteration 1 checkpoint directory (e.g. `MOODPO/outputs/iter1/checkpoint-XXXX`) and pass it as `--model_path`:

```bash
accelerate launch MOODPO/moodpo/moodpo.py \
  --wandb_name "moodpo_iter2" \
  --sft_model_path "<BASE_OR_SFT_MODEL_ID_OR_PATH>" \
  --model_path "/absolute/path/to/MOODPO/outputs/iter1/checkpoint-XXXX" \
  --csv_path "/absolute/path/to/prompts_train.csv" \
  --prompt_column "prompt" \
  --reward_models helpful harmless humor \
  --output_dir "MOODPO/outputs/iter2" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-7 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True
```

- **Reward models**: the `--reward_models` values are mapped inside `MOODPO/moodpo/moodpo.py` (see `REWARD_MODEL_PATHS`).
- **W&B logging**: this script uses `wandb`. If you do not want logging, set `--disable_wandb True` and remove `pip install wandb`.
- **OOM**: reduce `--batch_size
- **Summary dataset**: when training on the summary dataset, replace `OnlineDPOTrainer` with `OnlineDPOTrainer_Summary`
  (defined in `MOODPO/moodpo_FINAL/online_dpo_trainer_summary.py`).

**Iteration 1 (summary, fresh start)**:

```bash
accelerate launch MOODPO/moodpo/moodpo.py \
  --wandb_name "moodpo_summary_iter1" \
  --exp_type "summary" \
  --sft_model_path "<BASE_OR_SFT_MODEL_ID_OR_PATH>" \
  --csv_path "/absolute/path/to/summary_prompts_train.csv" \
  --prompt_column "prompt" \
  --reward_models summary faithful \
  --output_dir "MOODPO/outputs/summary/iter1" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-7 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True
```

**Iteration 2 (summary, resume)**:

```bash
accelerate launch MOODPO/moodpo/moodpo.py \
  --wandb_name "moodpo_summary_iter2" \
  --exp_type "summary" \
  --sft_model_path "<BASE_OR_SFT_MODEL_ID_OR_PATH>" \
  --model_path "/absolute/path/to/MOODPO/outputs/summary/iter1/checkpoint-XXXX" \
  --csv_path "/absolute/path/to/summary_prompts_train.csv" \
  --prompt_column "prompt" \
  --reward_models summary faithful \
  --output_dir "MOODPO/outputs/summary/iter2" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-7 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True
```