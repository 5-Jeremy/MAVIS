import random, torch
import os, yaml
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
import warnings
import argparse
from time import time_ns
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
from transformers import TrainingArguments
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
from peft import LoraConfig, get_peft_model

set_seed(16)

from utils.loading_utils import load_value_model, load_for_anthropic, load_for_summary
from utils.search_utils import parse_value_model_iter
from utils.at_utils import MixedTreeDataStructure
from utils.distill_utils import TokenRegressionTrainerND, TreeDataset_Distill, TokenRegressionCollatorND, MixedTreeDataset_Distill, freeze_llama_backbone

A = argparse.ArgumentParser()
A.add_argument("--dataset", type=str, choices=["anthropic", "summary"], default="anthropic")
A.add_argument("--data_dir", type=str, default=None, help="Directory containing the data for training; can be any data which is in the format used for training value models. To take data from multiple directories, list their paths in a .txt file and provide the path to the file.")
A.add_argument("--output_dir", type=str, default="Training_Output/distill")
A.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train the value model for. Default is 1.")
A.add_argument("--num_training_steps", type=int, default=None, help="Number of training batches to use (After this many batches, the learning rate will be zero). If not set, it will be automatically calculated based on the dataset size and batch size.")
A.add_argument("--init_checkpoint", type=str, default=None, help="Path to the checkpoint from which to initialize the value model. If none is set, it will be initialized from TinyLlama 1.1 with a randomly initialized value head.")
A.add_argument("--batch_size", type=int, default=32)
A.add_argument("--lr", type=float, default=4e-5, help="Learning rate for training the value model.")
A.add_argument("--weight_decay", type=float, default=2e-3, help="Weight decay for the optimizer.")
A.add_argument("--no_warmup", action="store_true", help="If set, the learning rate will not be warmed up at the beginning of training. This is useful for iterative training where the model has already been trained.")
A.add_argument("--grad_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps to use. Default is 1 (no accumulation).")
A.add_argument("--value_model_dir", type=str, default="Models/value_models/", help="Directory containing the value models for the objectives; it is assumed that the directory contains a subdirectory for each objective used, with the subdirectories containing the trained value model weights")
A.add_argument("--value_model_iter", type=str, default="0,0,0", help="The iteration of the value model to use; this is used to load the correct checkpoint from the value_model_dir")
args = A.parse_args()
data_dir = args.data_dir
output_dir = args.output_dir

args.no_warmup = True # No need for warmup when performing distillation

if args.dataset == "anthropic":
    loaded_assets = load_for_anthropic(include_gen_model=False, include_inputs=False, include_rewards=False, base_model_type="llama")
elif args.dataset == "summary":
    loaded_assets = load_for_summary(include_gen_model=False, include_inputs=False, include_rewards=False, base_model_type="llama")
tokenizer = loaded_assets["gen_tokenizer"]
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

from utils.gen_utils import LlamaValueModel
model = LlamaValueModel.from_pretrained("TinyLlama/TinyLlama_v1.1", num_labels=3, problem_type='regression', torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.model.padding_idx = tokenizer.pad_token_id
model.model.config.pad_token_id = tokenizer.pad_token_id

# We only want to add a new PEFT adapter if we are not loading from a checkpoint (i.e. we are training from scratch rather
# than doing iterative training)
if args.init_checkpoint is None:
    peft_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.2,
        bias="none",
        task_type="SEQ_CLS",
        # Include the QKV projections and the feed-forward layers
        target_modules=target_modules,
        # Also keep the classifier unfrozen
        modules_to_save=["score"],
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)

dataset_valid_objectives = ["help", "harm", "humor"] if args.dataset == "anthropic" else ["summarization", "faithful"]
iter_nums = parse_value_model_iter(args.value_model_iter, dataset_valid_objectives)
teacher_model_paths = [f"{args.value_model_dir}/iter_{iter}/{objective}" for objective, iter in iter_nums.items() if iter is not None]
teacher_models = [load_value_model(dataset="anthropic", checkpoint=path) for path in teacher_model_paths]

if args.data_dir.endswith(".txt"):
    # Load multiple data directories from a .txt file
    with open(args.data_dir, "r") as f:
        data_dirs = [line.strip() for line in f.readlines()]
else:
    data_dirs = [args.data_dir]

# Can add more directories to the list to mix datasets
mixed_tree_ds = MixedTreeDataStructure(data_dirs)

# Right now we don't bother with evaluation
def compute_metrics(eval_pred, inputs=None):
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    # Some models return (logits, ...); take first
    if isinstance(preds, tuple):
        preds = preds[0]

    # Expect shapes [B, L, D]
    assert preds.ndim == 3, f"preds should be [B, L, D], got {preds.shape}"
    assert labels.ndim == 3, f"labels should be [B, L, D], got {labels.shape}"
    B, L, D = preds.shape

    mse = 0
    # Optional: per-dimension metrics
    mse_per_dim = np.zeros(D)

    # Return plain Python floats
    out = {"mse": mse}
    # Also log per-dim (as separate scalars)
    out.update({f"mse_dim_{i}": float(v) for i, v in enumerate(mse_per_dim)})
    return out

dataset_tr = MixedTreeDataset_Distill(mixed_tree_ds, split="train", pad_token_id=tokenizer.pad_token_id, teacher_models=teacher_models)
dataset_val = MixedTreeDataset_Distill(mixed_tree_ds, split="val", pad_token_id=tokenizer.pad_token_id, teacher_models=teacher_models)

# Need to compute the number of training batches in order to set the learning rate scheduler
num_train_epochs = args.num_epochs
batch_size = args.batch_size
num_training_steps = (dataset_tr.__len__()//(batch_size*args.grad_accumulation_steps))*num_train_epochs if args.num_training_steps is None else args.num_training_steps

train_args = TrainingArguments(
    num_train_epochs=num_train_epochs,
    evaluation_strategy = "no", # Currently no evaluation
    save_strategy = "steps",
    save_steps = 50 if args.dataset == 'anthropic' else 200,
    save_total_limit=5,
    learning_rate=args.lr,
    lr_scheduler_type='linear',
    lr_scheduler_kwargs={'num_warmup_steps':100 if not args.no_warmup else 0, 'num_training_steps':num_training_steps},
    optim='adafactor',
    gradient_accumulation_steps=args.grad_accumulation_steps,
    per_device_train_batch_size=batch_size,
    report_to="tensorboard",
    logging_dir=output_dir,
    logging_steps=10,
    weight_decay=args.weight_decay,
    output_dir=output_dir,
    ddp_find_unused_parameters=False, # Recommended for performance
    bf16=True if args.dataset == 'summary' else False, # Use mixed precision training with summary dataset
    remove_unused_columns=False, # Need this because we use extra dataset features for the collator
    )

# Save all arguments to a yaml file
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "training_args.yaml"), "w") as f:
    yaml.dump(vars(train_args) | vars(args), f)
training_logfile = open(os.path.join(output_dir, "training_log.txt"), "w")
def log(message):
    print(message)
    training_logfile.write(message + "\n")
    training_logfile.flush()

from transformers.optimization import get_linear_schedule_with_warmup
class TrainerWithLinearWarmupSchedule(TokenRegressionTrainerND):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.lr_scheduler_kwargs["num_warmup_steps"], 
                num_training_steps=self.args.lr_scheduler_kwargs["num_training_steps"])
            self._created_lr_scheduler = True
        return self.lr_scheduler

train_start_time = time_ns()
from transformers import TrainerCallback
class MilestoneTimerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if state.global_step % train_args.eval_steps == 0:
                log(f"Reached step {state.global_step} after {(time_ns() - train_start_time)/1e9} seconds.")

trainer = TrainerWithLinearWarmupSchedule(
    model,
    train_args,
    train_dataset=dataset_tr,
    eval_dataset=None,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # callbacks=[MilestoneTimerCallback()], # TODO: bring this back when we have an eval dataset
    data_collator=TokenRegressionCollatorND(tokenizer, num_targets=3), # TODO: make num_targets dynamic
    # dimension_weights=[1.0, 0.5, 2.0],  # optional
)

trainer.train()
log(f"Phase 1 completed in {(time_ns() - train_start_time) / 1e9} seconds.")

freeze_llama_backbone(model)

phase2_num_training_steps = num_training_steps // 10
phase2_train_args = train_args
phase2_train_args.learning_rate = args.lr / 10.0
phase2_train_args.num_train_epochs = 1
phase2_train_args.per_device_train_batch_size = batch_size*2 # Double the batch size since memory is less of an issue in this phase
phase2_train_args.weight_decay = 0.0
phase2_train_args.max_steps = phase2_num_training_steps
phase2_train_args.lr_scheduler_kwargs={'num_warmup_steps':0, 'num_training_steps':phase2_num_training_steps}

phase2_trainer = TrainerWithLinearWarmupSchedule(
    model,
    phase2_train_args,
    train_dataset=dataset_tr,
    eval_dataset=None,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=TokenRegressionCollatorND(tokenizer, num_targets=3),
)

phase2_trainer.train()
log(f"Full training completed in {(time_ns() - train_start_time) / 1e9} seconds.")

# Save the model
model.save_pretrained(output_dir)
