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
from utils.at_utils import ValNodeTokensHDF5, TreeDataset_HDF5, TreeDataset_Soft
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

from utils.loading_utils import load_value_model, load_for_anthropic, load_for_summary

A = argparse.ArgumentParser()
A.add_argument("--dataset", type=str, choices=["anthropic", "summary"], default="anthropic")
A.add_argument("--objective", type=str, default="help")
A.add_argument("--data_dir", type=str, default="Anthropic/data_for_training/tokenized_5000/")
A.add_argument("--output_dir", type=str, default="training_output/")
A.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train the value model for. Default is 1.")
A.add_argument("--num_training_steps", type=int, default=None, help="Number of training batches to use (After this many batches, the learning rate will be zero). If not set, it will be automatically calculated based on the dataset size and batch size.")
A.add_argument("--init_checkpoint", type=str, default=None, help="Path to the checkpoint from which to initialize the value model. If none is set, it will be initialized from TinyLlama 1.1 with a randomly initialized value head.")
A.add_argument("--batch_size", type=int, default=32) # NOTE: default varies based on objective and is defined below
A.add_argument("--lr", type=float, default=4e-5, help="Learning rate for training the value model.")
A.add_argument("--weight_decay", type=float, default=2e-3, help="Weight decay for the optimizer.")
A.add_argument("--no_warmup", action="store_true", help="If set, the learning rate will not be warmed up at the beginning of training. This is useful for iterative training where the model has already been trained.")
A.add_argument("--grad_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps to use. Default is 1 (no accumulation).")
A.add_argument("--KL_penalty", type=float, default=None, help="KL divergence penalty multiplier. Using this argument will train a soft value model.")
args = A.parse_args()
data_dir = args.data_dir
output_dir = args.output_dir

if args.dataset == "anthropic":
    assert args.objective in ["help", "harm", "humor"], "Objective must be one of 'help', 'harm', or 'humor' for the Anthropic dataset."
    loaded_assets = load_for_anthropic(include_gen_model=False, include_inputs=False, include_rewards=False)
elif args.dataset == "summary":
    assert args.objective in ["summarization", "faithful"], "Objective must be one of 'summarization' or 'faithful' for the Summary dataset."
    loaded_assets = load_for_summary(include_gen_model=False, include_inputs=False, include_rewards=False)
tokenizer = loaded_assets["gen_tokenizer"]
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model = load_value_model(dataset=args.dataset, checkpoint=args.init_checkpoint, torch_dtype=torch.float32)
# We only want to add a new PEFT adapter if we are not loading from a checkpoint (i.e. we are training from scratch rather
# than doing iterative training)
if args.init_checkpoint is None:
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

# Extract the tree containing the training data. Note that the ValNodeTokensHDF5 class must be imported for this to work
# We need a dictionary mapping prompt names to their respective trees in order to easily access the tokens from the hdf5 file
train_roots = {}
for file in os.listdir(os.path.join(data_dir,"train")):
    if file.endswith(".pkl"):
        prompt_name = os.path.basename(file).split("_")[0]
        with open(os.path.join(data_dir,"train",file), "rb") as f:
            train_roots[prompt_name] = pickle.load(f)
val_roots = {}
for file in os.listdir(os.path.join(data_dir,"val")):
    if file.endswith(".pkl"):
        prompt_name = os.path.basename(file).split("_")[0]
        with open(os.path.join(data_dir,"val",file), "rb") as f:
            val_roots[prompt_name] = pickle.load(f)

tokens_file = os.path.join(data_dir, "all_tokens.hdf5")
assert os.path.exists(tokens_file), f"Tokens file {tokens_file} does not exist. Please ensure it is present in the data directory."

# For the harmlessness objective, when the data was generated using a MAVIS policy we do not want to remove any bottom nodes
# since the trees are already very shallow
fraction_bottom_nodes_to_keep = 1.0 if args.objective in ["harm"] and args.init_checkpoint is not None else 0.5
exclude_leaves = True 
if args.KL_penalty is None:
    dataset_tr = TreeDataset_HDF5(train_roots, tokens_file, fraction_bottom_nodes_to_keep=fraction_bottom_nodes_to_keep, 
                                objective=args.objective, pad_token_id=tokenizer.pad_token_id)
    dataset_val = TreeDataset_HDF5(val_roots, tokens_file, exclude_leaves=exclude_leaves, objective=args.objective, 
                                pad_token_id=tokenizer.pad_token_id, fraction_bottom_nodes_to_keep=fraction_bottom_nodes_to_keep)
else:
    # If KL_penalty is specified, we are training a soft value model
    dataset_tr = TreeDataset_Soft(train_roots, tokens_file, fraction_bottom_nodes_to_keep=fraction_bottom_nodes_to_keep, 
                                objective=args.objective, pad_token_id=tokenizer.pad_token_id, KL_penalty=args.KL_penalty)
    dataset_val = TreeDataset_Soft(val_roots, tokens_file, exclude_leaves=exclude_leaves, objective=args.objective, 
                                pad_token_id=tokenizer.pad_token_id, fraction_bottom_nodes_to_keep=fraction_bottom_nodes_to_keep,
                                KL_penalty=args.KL_penalty)

metric = load('mse')

# Need to compute the number of training batches in order to set the learning rate scheduler
num_train_epochs = args.num_epochs
batch_size = args.batch_size
num_training_steps = (dataset_tr.__len__()//(batch_size*args.grad_accumulation_steps))*num_train_epochs if args.num_training_steps is None else args.num_training_steps

train_args = TrainingArguments(
    num_train_epochs=num_train_epochs,
    evaluation_strategy = "steps",
    eval_steps = 50 if args.dataset == 'anthropic' else 200,
    save_strategy = "steps",
    save_steps = 50 if args.dataset == 'anthropic' else 200,
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
    bf16=True if args.dataset == 'summary' else False, # Use mixed precision training with summary dataset
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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)

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
