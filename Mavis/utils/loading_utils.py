# Utility functions for loading different combinations of models and datasets properly
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def get_dataset_from_objective(objective):
    if objective in ["help", "harm", "humor"]:
        return "anthropic"
    elif objective in ["summarization", "faithful"]:
        return "summary"
    elif objective in ["safeRLHF_help", "safeRLHF_harm"]:
        return "safeRLHF"
    else:
        raise ValueError(f"Objective {objective} is not supported. Supported objectives are: help, harm, humor, summarization, faithful.")

def load_sft_tokenizer(tokenizer_style):
    # We use the same tokenizer for both the 7B and 13B models. The Alpaca model uses a very similar tokenizer
    # except its pad token has a different form
    if tokenizer_style == "llama":
        if os.path.exists("Models/sft_model/llama_7b/anthropic"):
            path = "Models/sft_model/llama_7b/anthropic"
        elif os.path.exists("Models/sft_model/llama_13b/anthropic"):
            path = "Models/sft_model/llama_13b/anthropic"
        elif os.path.exists("Models/sft_model/llama_7b/summary"):
            path = "Models/sft_model/llama_7b/summary"
        elif os.path.exists("Models/sft_model/llama_13b/summary"):
            path = "Models/sft_model/llama_13b/summary"
        else:
            raise ValueError("Could not find an SFT model to load the tokenizer from.")
        gen_tokenizer = AutoTokenizer.from_pretrained(path)
    elif tokenizer_style == "alpaca":
        gen_tokenizer = AutoTokenizer.from_pretrained("PKU-Alignment/alpaca-7b-reproduced")
    else:
        raise ValueError(f"Unsupported tokenizer style. Should be either 'llama' or 'alpaca'.")
    return gen_tokenizer

# In this dataset, we have to get the prompt by removing the completion that already exists
def make_full_prompt_anthropic(x):
    x["prompt"] = x["chosen"][:x["chosen"].find("Assistant: ")+len("Assistant: ")]
    return x

def load_for_anthropic(csv_path="datasets/anthropic/", prompt_indices=[0], rewards=None, pre_tokenized=True, g_device="cuda:0", rm_device="cuda:0", 
                       split="test", include_gen_model=True, gen_model_name=None, include_rewards=True, include_inputs=True, 
                       obj_weights=None, ppo=False, morlhf=False, moodpo=False, base_model_type="llama_7b"):
    reward_model_dtype = torch.float32
    ret = {}
    ret["max_completion_len"] = 128
    if base_model_type == "llama":
        # If we only need the tokenizer, no need to specify 7b or 13b since they use the same one
        gen_tokenizer = load_sft_tokenizer("llama")
    else:
        assert base_model_type in ["llama_7b", "llama_13b"], "base_model_type must be one of llama_7b, llama_13b"
        gen_tokenizer = AutoTokenizer.from_pretrained(f"Models/sft_model/{base_model_type}/anthropic")
    ret["gen_tokenizer"] = gen_tokenizer
    if include_gen_model:
        if ppo or morlhf or moodpo:
            if obj_weights is not None:
                if moodpo:
                    # For MOODPO, the same model is used for any objective weight setting since the conditioning is
                    # done via the prompt
                    print(f"Using multi-objective online DPO with objective weights {obj_weights}")
                    gen_model_name = f"Models/moodpo/{base_model_type}/anthropic/"
                # The Models/morlhf directory contains all models fine-tuned for a specific weighting of objectives; the reward_soup
                # directory contains models that come from parameter-merging the single-objective Models/morlhf models
                elif obj_weights['help'] == 1.0 or obj_weights['harm'] == 1.0 or obj_weights['humor'] == 1.0:
                    if obj_weights['help'] == 1.0:
                        obj = 'help'
                    elif obj_weights['harm'] == 1.0:
                        obj = 'harm'
                    elif obj_weights['humor'] == 1.0:
                        obj = 'humor'
                    print(f"Using MORLHF for single objective: {obj}")
                    gen_model_name = f"Models/morlhf/{base_model_type}/anthropic/single/{obj}"
                elif morlhf:
                    if obj_weights['help'] == 0.0 or obj_weights['help'] is None:
                        obj_dir = "harm_humor"
                        main_weight = obj_weights['harm']
                    elif obj_weights['harm'] == 0.0 or obj_weights['harm'] is None:
                        obj_dir = "help_humor"
                        main_weight = obj_weights['help']
                    elif obj_weights['humor'] == 0.0 or obj_weights['humor'] is None:
                        obj_dir = "help_harm"
                        main_weight = obj_weights['help']
                    else:
                        raise ValueError("Cannot assign weight to all three objectives at once if using MORLHF")
                    print(f"Using MORLHF with objective weights {obj_weights}")
                    gen_model_name = f"Models/morlhf/{base_model_type}/anthropic/{obj_dir}/morlhf_{main_weight}"
                else:
                    # For reward soup, we allow for mixing three objectives at once
                    if obj_weights['help'] == 0.0 or obj_weights['help'] is None:
                        obj_dir = "harm_humor"
                        weight = obj_weights['harm']
                    elif obj_weights['harm'] == 0.0 or obj_weights['harm'] is None:
                        obj_dir = "help_humor"
                        weight = obj_weights['help']
                    elif obj_weights['humor'] == 0.0 or obj_weights['humor'] is None:
                        obj_dir = "help_harm"
                        weight = obj_weights['help']
                    else:
                        obj_dir = "help_harm_humor"
                        weight = ",".join([str(obj_weights['help']), str(obj_weights['harm']), str(obj_weights['humor'])])
                    print(f"Using reward soup with objective weights {obj_weights}")
                    gen_model_name = f"Models/reward_soup/{base_model_type}/anthropic/{obj_dir}/reward_soup_{weight}"
            else:
                raise ValueError("If using PPO model, must specify objective weights")
            if "adapter_config.json" in os.listdir(gen_model_name):
                print(f"Loading PEFT adapter from {gen_model_name}")
                sft_model = AutoModelForCausalLM.from_pretrained(f"Models/sft_model/{base_model_type}/anthropic",torch_dtype=torch.bfloat16, device_map=g_device)
                gen_model = PeftModel.from_pretrained(sft_model, gen_model_name, torch_dtype=torch.bfloat16, device_map=g_device)
            else:
                # If the peft adapter was merged with the SFT model weights
                print(f"Loading merged model from {gen_model_name}")
                gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, torch_dtype=torch.bfloat16, device_map=g_device)
        else:
            gen_model = AutoModelForCausalLM.from_pretrained(f"Models/sft_model/{base_model_type}/anthropic",torch_dtype=torch.bfloat16, device_map=g_device)
        eos_token_id = gen_tokenizer.eos_token_id
        ret["gen_model"] = gen_model
        ret["eos_token_id"] = eos_token_id
    if include_rewards:
        # This part is used for label_tree.py
        if type(rewards) == str:
            if rewards == "all":
                rewards = ["help", "harm", "humor"]
            else:
                rewards = [rewards]
        ORM_models, ORM_tokenizers = {}, {}
        if "help" in rewards:
            ORM_model = AutoModelForSequenceClassification.from_pretrained("Ray2333/gpt2-large-helpful-reward_model", local_files_only=False, torch_dtype=reward_model_dtype).to(rm_device)
            ORM_tokenizer = AutoTokenizer.from_pretrained("Ray2333/gpt2-large-helpful-reward_model")
            ORM_tokenizer.pad_token_id = ORM_tokenizer.eos_token_id
            ORM_model.config.pad_token_id = ORM_tokenizer.pad_token_id
            ORM_models["help"] = ORM_model
            ORM_tokenizers["help"] = ORM_tokenizer
        if "harm" in rewards:
            ORM_model = AutoModelForSequenceClassification.from_pretrained("Ray2333/gpt2-large-harmless-reward_model", local_files_only=False, torch_dtype=reward_model_dtype).to(rm_device)
            ORM_tokenizer = AutoTokenizer.from_pretrained("Ray2333/gpt2-large-harmless-reward_model")
            ORM_tokenizer.pad_token_id = ORM_tokenizer.eos_token_id
            ORM_model.config.pad_token_id = ORM_tokenizer.pad_token_id
            ORM_models["harm"] = ORM_model
            ORM_tokenizers["harm"] = ORM_tokenizer
        if "humor" in rewards:
            # This reward model is based on distilbert-base-uncased, so it has a pad token but no eos token
            ORM_model = AutoModelForSequenceClassification.from_pretrained("mohameddhiab/humor-no-humor", local_files_only=False, torch_dtype=reward_model_dtype, num_labels=2).to(rm_device)
            ORM_tokenizer = AutoTokenizer.from_pretrained("mohameddhiab/humor-no-humor")
            ORM_models["humor"] = ORM_model
            ORM_tokenizers["humor"] = ORM_tokenizer
        assert len(list(ORM_models.keys())) > 0, "No ORM models were loaded. Please check the rewards parameter."
        assert list(ORM_models.keys()) == list(ORM_tokenizers.keys()), "ORM models and tokenizers do not have the same keys."
        ret["ORM_model"] = ORM_models
        ret["ORM_tokenizer"] = ORM_tokenizers
        def get_rewards(seq):
            outputs = {}
            with torch.no_grad():
                for k, v in ORM_models.items():
                    # We pass the prompt to the humor reward model (our PPO code does the same)
                    tokenized = ORM_tokenizers[k](seq, return_tensors="pt", padding=True, padding_side="right")
                    if tokenized['input_ids'].shape[1] > 512:
                        print(f"Warning: Input sequence is too long for {k} ORM model. Truncating to first 512 tokens.")
                        tokenized['input_ids'] = tokenized['input_ids'][:, :512]
                        tokenized['attention_mask'] = tokenized['attention_mask'][:, :512]
                    inputs = tokenized['input_ids'].to(rm_device)
                    attn_mask = tokenized['attention_mask'].to(rm_device)
                    if k == "humor":
                        outputs[k] = v(inputs, attention_mask=attn_mask).logits[:,1].cpu().detach()
                    else:
                        outputs[k] = v(inputs, attention_mask=attn_mask).logits.cpu().detach()
            return outputs
        ret["get_rewards"] = get_rewards
    if include_inputs:
        if csv_path is None:
            dataset = load_dataset("Anthropic/hh-rlhf", split=split)
            dataset = dataset.map(make_full_prompt_anthropic)
            # Filter out prompts that are too long
            dataset = dataset.filter(lambda x: len(gen_tokenizer(x["prompt"])["input_ids"]) <= 200)
        else:
            # We have a separate CSV file for each split, but each CSV file has its data organized by split; specifically, there is only
            # one split in the CSV file, and it is called "train"
            dataset = load_dataset("csv", data_files=os.path.join(csv_path, f"anthropic_{split}_deduped.csv"))['train']
            # Note that the data in the CSV file should already have been filtered for length
        # Pre-tokenize if desired
        if pre_tokenized:
            dataset = dataset.map(lambda x: gen_tokenizer(x["prompt"], return_tensors="pt"), batched=False)
        # Get prompts corresonding to the desired rows of the dataset
        ret["prompts"] = dataset[prompt_indices]["prompt"]
    return ret

from transformers import LlamaForSequenceClassification
from utils.gen_utils import LlamaValueModel
# Use this for the tinyllama model
def load_value_model(checkpoint=None, device="cuda:0", torch_dtype="auto", tokenizer=None, dataset="anthropic", num_objectives=1):
    if tokenizer is None:
        if dataset in ['anthropic', 'summary']:
            tokenizer = load_sft_tokenizer('llama')
        elif dataset == 'safeRLHF':
            tokenizer = load_sft_tokenizer('alpaca')
        else:
            raise ValueError("Unsupported dataset")

    if checkpoint is None:
        model = LlamaForSequenceClassification.from_pretrained("TinyLlama/TinyLlama_v1.1", num_labels=num_objectives, problem_type='regression', torch_dtype=torch_dtype)
        # When making changes after the model has been initialized, we need to make sure that the changes are applied to the 
        # underlying LlamaModel as well
        model.model.resize_token_embeddings(len(tokenizer))
    else:
        # Assume we are loading a LoRa adapter
        base_model = LlamaValueModel.from_pretrained("TinyLlama/TinyLlama_v1.1", num_labels=num_objectives, problem_type='regression', torch_dtype=torch_dtype)
        # If loading from a checkpoint, the pad token should already be set correctly in the config
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, checkpoint, torch_dtype=torch_dtype, is_trainable=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.model.padding_idx = tokenizer.pad_token_id
    model.model.config.pad_token_id = tokenizer.pad_token_id
    if device is not None: # When using accelerate, we don't want to set the model device manually
        model = model.to(device)
    return model

#######################################
def make_full_prompt_summary(x):
    info_post = x["info"]["post"].replace("\n", " ")
    x["prompt"] = f"### Instruction: Generate a one-sentence summary of this post. ### Input: {info_post} ### Response: "
    return {"prompt": x["prompt"]}

def load_for_summary(csv_path="datasets/summary/", prompt_indices=[0], rewards=None, pre_tokenized=True, g_device="cuda:0", rm_device="cuda:0", 
                       split="test", include_gen_model=True, gen_model_name=None, include_rewards=True, include_inputs=True, 
                       obj_weights=None, ppo=False, morlhf=False, moodpo=False, base_model_type="llama_7b", rescale=False):
    reward_model_dtype = torch.float32
    ret = {}
    ret["max_completion_len"] = 48
    if base_model_type == "llama":
        # If we only need the tokenizer, no need to specify 7b or 13b since they use the same one
        gen_tokenizer = load_sft_tokenizer("llama")
    else:
        assert base_model_type in ["llama_7b", "llama_13b"], "base_model_type must be one of llama_7b, llama_13b"
        gen_tokenizer = AutoTokenizer.from_pretrained(f"Models/sft_model/{base_model_type}/summary")
    ret["gen_tokenizer"] = gen_tokenizer
    if include_gen_model:
        if ppo or morlhf or moodpo:
            if obj_weights is not None:
                if moodpo:
                    # For MOODPO, the same model is used for any objective weight setting since the conditioning is
                    # done via the prompt
                    print(f"Using multi-objective online DPO with objective weights {obj_weights}")
                    gen_model_name = f"Models/moodpo/{base_model_type}/summary/"
                # The Models/morlhf directory contains all models fine-tuned for a specific weighting of objectives; the reward_soup
                # directory contains models that come from parameter-merging the single-objective Models/morlhf models
                elif obj_weights['summarization'] == 1.0 or obj_weights['faithful'] == 1.0:
                    if obj_weights['summarization'] == 1.0:
                        obj = 'summarization'
                    elif obj_weights['faithful'] == 1.0:
                        obj = 'faithful'
                    print(f"Using MORLHF for single objective: {obj}")
                    gen_model_name = f"Models/morlhf/{base_model_type}/summary/single/{obj}"
                elif morlhf:
                    # We treat summarization as the 'main objective' and pick the model based on that weight
                    print(f"Using MORLHF with objective weights {obj_weights}")
                    gen_model_name = f"Models/morlhf/{base_model_type}/summary/morlhf_{obj_weights['summarization']}"
                else:
                    print(f"Using reward soup with objective weights {obj_weights}")
                    gen_model_name = f"Models/reward_soup/{base_model_type}/summary/reward_soup_{obj_weights['summarization']}"
            else:
                raise ValueError("If using PPO model, must specify objective weights")
            if "adapter_config.json" in os.listdir(gen_model_name):
                print(f"Loading PEFT adapter from {gen_model_name}")
                sft_model = AutoModelForCausalLM.from_pretrained(f"Models/sft_model/{base_model_type}/summary",torch_dtype=torch.bfloat16, device_map=g_device)
                gen_model = PeftModel.from_pretrained(sft_model, gen_model_name, torch_dtype=torch.bfloat16, device_map=g_device)
            else:
                # If the peft adapter was merged with the SFT model weights
                print(f"Loading merged model from {gen_model_name}")
                gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, torch_dtype=torch.bfloat16, device_map=g_device)
        else:
            gen_model = AutoModelForCausalLM.from_pretrained(f"Models/sft_model/{base_model_type}/summary",torch_dtype=torch.bfloat16, device_map=g_device)
        eos_token_id = gen_tokenizer.eos_token_id
        ret["gen_model"] = gen_model
        ret["eos_token_id"] = eos_token_id
    if include_rewards:
        if rescale and base_model_type == "llama_13b":
            faith_scale_factor = 2.0
        else:
            faith_scale_factor = 1.0
        if type(rewards) == str:
            if rewards == "all":
                rewards = ["summarization", "faithful"]
            else:
                rewards = [rewards]
        ORM_models, ORM_tokenizers = {}, {}
        if "summarization" in rewards:
            ORM_model = AutoModelForSequenceClassification.from_pretrained("Tristan/gpt2_reward_summarization", local_files_only=False, torch_dtype=reward_model_dtype).to(rm_device)
            ORM_tokenizer = AutoTokenizer.from_pretrained("Tristan/gpt2_reward_summarization")
            ORM_tokenizer.pad_token_id = ORM_tokenizer.eos_token_id
            ORM_model.config.pad_token_id = ORM_tokenizer.pad_token_id
            ORM_models["summarization"] = ORM_model
            ORM_tokenizers["summarization"] = ORM_tokenizer
        if "faithful" in rewards:
            # This reward model is similar to the humor reward model for the anthropic dataset
            # Note that the warning about num_labels when loading the model is to be expected
            ORM_model = AutoModelForSequenceClassification.from_pretrained("CogComp/bart-faithful-summary-detector", local_files_only=False, torch_dtype=reward_model_dtype).to(rm_device)
            ORM_tokenizer = AutoTokenizer.from_pretrained("CogComp/bart-faithful-summary-detector")
            ORM_models["faithful"] = ORM_model
            ORM_tokenizers["faithful"] = ORM_tokenizer
        assert len(list(ORM_models.keys())) > 0, "No ORM models were loaded. Please check the rewards parameter."
        assert list(ORM_models.keys()) == list(ORM_tokenizers.keys()), "ORM models and tokenizers do not have the same keys."
        ret["ORM_model"] = ORM_models
        ret["ORM_tokenizer"] = ORM_tokenizers
        def get_rewards(seq):
            # Need to preprocess the text
            posts = []
            posts_with_tags = []
            generated_summaries = []
            for _seq in seq:
                # For the faithful reward model, we remove the "### Input: " and "### Response: " tags and manually add the seperator
                assert "### Input: " in _seq and "### Response: " in _seq
                posts_with_tags.append(_seq.split("### Response:")[0] + "### Response:")
                posts.append(_seq.split("### Input: ")[-1].split("### Response:")[0].strip())
                generated_summaries.append(_seq.split("### Response: ")[-1].strip())
            outputs = {}
            with torch.no_grad():
                for k, v in ORM_models.items():
                    ### Perform tokenization
                    if k == "faithful":
                        texts_for_rm = [generated_summaries[i] + "</s></s>" + posts[i] for i in range(len(posts))]
                        tokenized = ORM_tokenizers[k](texts_for_rm, return_tensors="pt", padding=True, padding_side="right", truncation=True, max_length=1024)
                        inputs = tokenized['input_ids'].to(rm_device)
                        attn_mask = tokenized['attention_mask'].to(rm_device)
                        outputs[k] = faith_scale_factor*v(inputs, attention_mask=attn_mask).logits[:,1].cpu().detach()
                    else:
                        bos_token = ORM_tokenizers[k].bos_token
                        texts_for_rm = [generated_summaries[i] + ' ' + bos_token + ' ' + posts[i] for i in range(len(posts))]
                        tokenized = ORM_tokenizers[k](texts_for_rm, return_tensors="pt", padding=True, padding_side="right")
                        inputs = tokenized['input_ids'].to(rm_device)
                        attn_mask = tokenized['attention_mask'].to(rm_device)
                        outputs[k] = v(inputs, attention_mask=attn_mask).logits.cpu().detach()
            return outputs
        ret["get_rewards"] = get_rewards
    if include_inputs:
        if csv_path is None:
            dataset = load_dataset("openai/summarize_from_feedback", split=split)
            # Filter out prompts that are too long
            dataset = dataset.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200)
            dataset = dataset.map(make_full_prompt_summary)
            # Filter out prompts that are longer than 512 tokens or shorter than 8 tokens
            dataset = dataset.filter(lambda x: len(gen_tokenizer(x["prompt"])["input_ids"]) <= 512)
            dataset = dataset.filter(lambda x: len(gen_tokenizer(x["prompt"])["input_ids"]) >= 8)
        else:
            # We have a separate CSV file for each split, but each CSV file has its data organized by split; specifically, there is only
            # one split in the CSV file, and it is called "train"
            dataset = load_dataset("csv", data_files=os.path.join(csv_path, f"summary_{split}_deduped.csv"))['train']
            # Note that the data in the CSV file should already have been filtered for length
        # Pre-tokenize if desired
        if pre_tokenized:
            dataset = dataset.map(lambda x: gen_tokenizer(x["prompt"], return_tensors="pt"), batched=False)
        # Get prompts corresonding to the desired rows of the dataset
        ret["prompts"] = dataset[prompt_indices]["prompt"]
    return ret

#######################################
# This comes from the PARM code (https://github.com/Baijiong-Lin/PARM/tree/main)
def format_prompt_pku_safe_rlhf(
    input: str,  
    eos_token: str,
) -> str:
    PROMPT_BEGIN_pku_safe_rlhf: str = 'BEGINNING OF CONVERSATION: '
    PROMPT_USER_pku_safe_rlhf: str = 'USER: {input} '
    PROMPT_ASSISTANT_pku_safe_rlhf: str = 'ASSISTANT:'  # should not have a space at the end
    
    assert isinstance(input, str), f'Unsupported type of `input`: {type(input)}. Expected: str.' 

    if isinstance(input, str):
        input = [input]
    elif not isinstance(input, list):
        raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str or list[str].')

    if len(input) % 2 != 1:
        raise ValueError(
            'The length of `input` must be odd, while `input` must end at the user question.',
        )

    buffer = [PROMPT_BEGIN_pku_safe_rlhf]
    for i, line in enumerate(input):
        if i % 2 == 0:
            # User input
            buffer.extend((PROMPT_USER_pku_safe_rlhf.format(input=line), PROMPT_ASSISTANT_pku_safe_rlhf))
        else:
            # Assistant response
            buffer.extend((line, eos_token))

    return ''.join(buffer)

# Right now we don't have finetuned models for safeRLHF
def load_for_safeRLHF(csv_path="datasets/safeRLHF/", prompt_indices=[0], rewards=None, pre_tokenized=True, g_device="cuda:0", rm_device="cuda:0", 
                       split="test", include_gen_model=True, gen_model_name=None, include_rewards=True, include_inputs=True, 
                       obj_weights=None):
    ret = {}
    ret["max_completion_len"] = 512
    gen_tokenizer = load_sft_tokenizer('alpaca')
    ret["gen_tokenizer"] = gen_tokenizer
    if include_gen_model:
        gen_model = AutoModelForCausalLM.from_pretrained("PKU-Alignment/alpaca-7b-reproduced",torch_dtype=torch.bfloat16, device_map=g_device)
        eos_token_id = gen_tokenizer.eos_token_id
        ret["gen_model"] = gen_model
        ret["eos_token_id"] = eos_token_id
    if include_rewards:
        from safe_rlhf.models import AutoModelForScore
        # This part is used for label_tree.py
        if type(rewards) == str:
            if rewards == "all":
                rewards = ["safeRLHF_help", "safeRLHF_harm"]
            else:
                rewards = [rewards]
        ORM_models, ORM_tokenizers = {}, {}
        if "safeRLHF_help" in rewards:
            ORM_model = AutoModelForScore.from_pretrained('PKU-Alignment/beaver-7b-v1.0-reward', torch_dtype=torch.bfloat16, device_map=rm_device)
            ORM_model.eval()
            ORM_tokenizer = AutoTokenizer.from_pretrained("PKU-Alignment/beaver-7b-v1.0-reward")
            ORM_models["safeRLHF_help"] = ORM_model
            ORM_tokenizers["safeRLHF_help"] = ORM_tokenizer
        if "safeRLHF_harm" in rewards:
            ORM_model = AutoModelForScore.from_pretrained('PKU-Alignment/beaver-7b-v1.0-cost', torch_dtype=torch.bfloat16, device_map=rm_device)
            ORM_model.eval()
            ORM_tokenizer = AutoTokenizer.from_pretrained("PKU-Alignment/beaver-7b-v1.0-cost")
            ORM_models["safeRLHF_harm"] = ORM_model
            ORM_tokenizers["safeRLHF_harm"] = ORM_tokenizer
        assert len(list(ORM_models.keys())) > 0, "No ORM models were loaded. Please check the rewards parameter."
        assert list(ORM_models.keys()) == list(ORM_tokenizers.keys()), "ORM models and tokenizers do not have the same keys."
        ret["ORM_model"] = ORM_models
        ret["ORM_tokenizer"] = ORM_tokenizers
        def get_rewards(seq):
            outputs = {}
            with torch.no_grad():
                # Reward models have same tokenizer, so use the first one in the list
                inputs = list(ORM_tokenizers.values())[0](seq, return_tensors='pt', padding=True, padding_side="right").to(rm_device)
                for k, v in ORM_models.items():
                    outputs[k] = v(**inputs)['end_scores'].cpu().detach()
                    if k == "safeRLHF_harm": # The model outputs a cost (harmfulness) which we convert to a reward
                        outputs[k] = -1*outputs[k]
            return outputs
        ret["get_rewards"] = get_rewards
    if include_inputs:
        # We have a separate CSV file for each split, but each CSV file has its data organized by split; specifically, there is only
        # one split in the CSV file, and it is called "train"
        dataset = load_dataset("csv", data_files=os.path.join(csv_path, f"safeRLHF_{split}_deduped.csv"))['train']
        # Pre-tokenize if desired
        if pre_tokenized:
            dataset = dataset.map(lambda x: gen_tokenizer(x["prompt"], return_tensors="pt"), batched=False)
        # Get prompts corresonding to the desired rows of the dataset
        ret["prompts"] = dataset[prompt_indices]["prompt"]
    return ret

import random
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
