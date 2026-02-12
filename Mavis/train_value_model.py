import random, torch
import os, yaml
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from h5py import File
import warnings
import argparse
from time import time_ns
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
from transformers import TrainingArguments, Trainer
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
from evaluate import load
from peft import LoraConfig, get_peft_model

set_seed(16)

from utils.loading_utils import load_value_model, load_for_anthropic, load_for_summary, load_for_safeRLHF
from utils.hdf5_utils import load_all_pickled_objects
from utils.at_utils import ValNodeTokensHDF5, TreeDataset_HDF5

A = argparse.ArgumentParser()
A.add_argument("--dataset", type=str, choices=["anthropic", "summary", "safeRLHF"], default="anthropic")
A.add_argument("--objective", type=str, default="help")
A.add_argument("--data_file", type=str, default="Training_Output/anthropic/data_for_training/tokenized_5000/all_data_new_format.hdf5")
A.add_argument("--output_dir", type=str, default="training_output/")
A.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train the value model for. Default is 1.")
A.add_argument("--num_training_steps", type=int, default=None, help="Number of training batches to use (After this many batches, the learning rate will be zero). If not set, it will be automatically calculated based on the dataset size and batch size.")
A.add_argument("--init_checkpoint", type=str, default=None, help="Path to the checkpoint from which to initialize the value model. If none is set, it will be initialized from TinyLlama 1.1 with a randomly initialized value head.")
A.add_argument("--batch_size", type=int, default=32)
A.add_argument("--lr", type=float, default=2e-5, help="Learning rate for training the value model.")
A.add_argument("--weight_decay", type=float, default=2e-3, help="Weight decay for the optimizer.")
A.add_argument("--no_warmup", action="store_true", help="If set, the learning rate will not be warmed up at the beginning of training. This is useful for iterative training where the model has already been trained.")
A.add_argument("--grad_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps to use. Default is 1 (no accumulation).")
A.add_argument("--KL_penalty", type=float, default=None, help="KL divergence penalty multiplier. Using this argument will train a soft value model.")
A.add_argument("--fraction_bottom_nodes_to_keep", type=float, default=None, help="Fraction of bottom nodes to keep in the training data. If not set, defaults to 0.5 for help and humor objectives and 1.0 for harm objective when doing iterative training (i.e. when --init_checkpoint is set).")
A.add_argument("--num_trees", type=int, default=None, help="If set, limits the total number of trees used for training data.")
A.add_argument("--num_val_trees", type=int, default=None, help="Number of trees to use for validation. If not set, defaults to 10% of the dataset.")
A.add_argument("--disable_tqdm", action="store_true", help="If set, disables the tqdm progress bars during training.")
args = A.parse_args()
output_dir = args.output_dir

if args.dataset == "anthropic":
    assert args.objective in ["help", "harm", "humor"], "Objective must be one of 'help', 'harm', or 'humor' for the Anthropic dataset."
    loaded_assets = load_for_anthropic(include_gen_model=False, include_inputs=False, include_rewards=False, base_model_type="llama")
elif args.dataset == "summary":
    assert args.objective in ["summarization", "faithful"], "Objective must be one of 'summarization' or 'faithful' for the Summary dataset."
    loaded_assets = load_for_summary(include_gen_model=False, include_inputs=False, include_rewards=False, base_model_type="llama")
elif args.dataset == "safeRLHF":
    assert args.objective in ["safeRLHF_help", "safeRLHF_harm"], "Objective must be one of 'safeRLHF_help' or 'safeRLHF_harm' for the safeRLHF dataset."
    loaded_assets = load_for_safeRLHF(include_gen_model=False, include_inputs=False, include_rewards=False)
tokenizer = loaded_assets["gen_tokenizer"]
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model = load_value_model(dataset=args.dataset, checkpoint=args.init_checkpoint, torch_dtype=torch.float32, 
                         tokenizer=tokenizer, num_objectives=1)

# We only want to add a new PEFT adapter if we are not loading from a checkpoint (i.e. we are training from scratch rather
# than doing iterative training)
if args.init_checkpoint is None and not args.no_lora:
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
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

data_file = File(args.data_file, "r")

if args.num_trees is not None and args.num_val_trees is not None:
    assert args.num_trees > args.num_val_trees, "If both num_trees and num_val_trees are set, num_trees must be larger than num_val_trees."

# Extract the tree containing the training data. Note that the ValNodeTokensHDF5 class must be imported for this to work
# We need a dictionary mapping prompt names to their respective trees in order to easily access the tokens from the hdf5 file
all_roots = load_all_pickled_objects(data_file)
get_prompt_indx = lambda name: int(name.replace("prompt", ""))
# We arbitrarily choose to use the prompts with the lowest indices for validation. To determine the range of indices that will
# go into the validation set, we first sort the list of names and then take the first num_val_trees of them.
all_prompt_indices = sorted([get_prompt_indx(name) for name in all_roots.keys()])
if args.num_trees is not None:
    assert len(all_prompt_indices) >= args.num_trees, "Not enough trees in the dataset to satisfy num_trees."
    all_prompt_indices = all_prompt_indices[:args.num_trees]
num_val_trees = args.num_val_trees if args.num_val_trees is not None else int(len(all_prompt_indices) * 0.1)
val_tree_indices = all_prompt_indices[:num_val_trees]
max_val_prompt_index = val_tree_indices[-1]
max_train_prompt_index = all_prompt_indices[-1]
train_roots = {k: v for k,v in all_roots.items() if get_prompt_indx(k) > max_val_prompt_index and get_prompt_indx(k) <= max_train_prompt_index}
val_roots = {k: v for k,v in all_roots.items() if get_prompt_indx(k) in range(0, max_val_prompt_index+1)}

tokens_file = data_file  # The tokens are stored in the same hdf5 file as the trees

# For the harmlessness objective, when the data was generated using a MAVIS policy we do not want to remove any bottom nodes
# since the trees are already very shallow
if args.fraction_bottom_nodes_to_keep is not None:
    fraction_bottom_nodes_to_keep = args.fraction_bottom_nodes_to_keep
else:
    fraction_bottom_nodes_to_keep = 1.0 if args.objective in ["harm", "safeRLHF_harm"] and args.init_checkpoint is not None else 0.4

dataset_tr = TreeDataset_HDF5(train_roots, args.data_file, fraction_bottom_nodes_to_keep=fraction_bottom_nodes_to_keep, 
                            objective=args.objective, pad_token_id=tokenizer.pad_token_id, KL_penalty=args.KL_penalty)
dataset_val = TreeDataset_HDF5(val_roots, args.data_file, exclude_leaves=True, objective=args.objective, 
                            pad_token_id=tokenizer.pad_token_id, fraction_bottom_nodes_to_keep=fraction_bottom_nodes_to_keep, 
                            KL_penalty=args.KL_penalty)

# Need to compute the number of training batches in order to set the learning rate scheduler
num_train_epochs = args.num_epochs
batch_size = args.batch_size
num_training_steps = (dataset_tr.__len__()//(batch_size*args.grad_accumulation_steps))*num_train_epochs if args.num_training_steps is None else args.num_training_steps

train_args = TrainingArguments(
    num_train_epochs=num_train_epochs,
    evaluation_strategy = "steps",
    eval_steps = 50 if args.dataset in ["anthropic", "safeRLHF"] else 200,
    save_strategy = "steps",
    save_steps = 50 if args.dataset in ["anthropic", "safeRLHF"] else 200,
    save_total_limit=5,
    load_best_model_at_end=True,
    learning_rate=args.lr,
    lr_scheduler_type='linear',
    lr_scheduler_kwargs={'num_warmup_steps':100 if not args.no_warmup else 0, 'num_training_steps':num_training_steps},
    optim='adafactor',
    gradient_accumulation_steps=args.grad_accumulation_steps,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=100,
    report_to="tensorboard",
    logging_dir=output_dir,
    logging_steps=10,
    weight_decay=args.weight_decay,
    output_dir=output_dir,
    metric_for_best_model='eval_loss',
    ddp_find_unused_parameters=False, # Recommended for performance
    bf16=True, # Use mixed precision training (unlikely to make a big difference)
    disable_tqdm=args.disable_tqdm,
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
log(f"Fraction of bottom nodes to keep: {fraction_bottom_nodes_to_keep}")

metric = load('mse')
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    return metric.compute(predictions=preds, references=labels)

from transformers.optimization import get_linear_schedule_with_warmup
class TrainerWithLinearWarmupSchedule(Trainer):
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
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[MilestoneTimerCallback()],
)

trainer.train()
log(f"Training completed in {(time_ns() - train_start_time) / 1e9} seconds.")

# Save the model (this should be the best one since we used load_best_model_at_end=True)
model.save_pretrained(output_dir)
