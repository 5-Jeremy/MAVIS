import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser
from peft import PeftModel
import numpy as np
import pandas as pd
from utils import load_main_tokenizer
tqdm.pandas()


@dataclass
class ScriptArguments:
    # save_directory: Optional[str] = field(default='13B_anthropic_sft')
    base_model_name: Optional[str] = field(default='meta-llama/Llama-2-7b-hf', metadata={"help": "local path to the base model or the huggingface id"})
    lora_name: Optional[str] = field(default='sft_model/summary', metadata={"help": "local path to the lora model"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
lora_name = script_args.lora_name
base_model_name = script_args.base_model_name
tokenizer_name = base_model_name # we use the same tokenizer for the base model
print('base model: ', base_model_name)
save_directory = os.path.basename(script_args.base_model_name) + '_' + os.path.basename(script_args.lora_name)
os.makedirs(os.path.join(save_directory), exist_ok=True)

accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

tokenizer = load_main_tokenizer(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.bfloat16, 
    device_map=gpu_id, 
)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, lora_name)
model = model.merge_and_unload() # merge lora weights


print("Saving last checkpoint of the model")
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
    

