import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, AutoTokenizer
from peft import PeftModel
tqdm.pandas()

def load_main_tokenizer(tokenizer_name):
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>" 
    DEFAULT_UNK_TOKEN = "<unk>" 

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = True)
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tokenizer

@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./sft_merged/')
    base_model_name: Optional[str] = field(default='meta-llama/Llama-2-7b-hf', metadata={"help": "local path to the base model or the huggingface id"})
    lora_name: Optional[str] = field(default='SFT_1', metadata={"help": "local path to the lora model"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
lora_name = script_args.lora_name
base_model_name = script_args.base_model_name
tokenizer_name = base_model_name # we use the same tokenizer for the base model
print('base model: ', base_model_name)
os.makedirs(os.path.join(script_args.save_directory), exist_ok=True)

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

save_path = script_args.save_directory
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)