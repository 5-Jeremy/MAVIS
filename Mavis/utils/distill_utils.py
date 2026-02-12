import torch
from torch.utils.data import Dataset
import numpy as np
from utils.at_utils import ValNodeTokensHDF5, TreeDataset_HDF5, MixedTreeDataStructure
from transformers import Trainer
import anytree as at
from h5py import File
from random import shuffle

class MixedTreeDataset_Distill(Dataset):
    def __init__(self, mixed_tree_ds: MixedTreeDataStructure, split: str, pad_token_id=32000, teacher_models=[]):
        assert len(teacher_models) > 0, "At least one teacher model must be provided"
        assert split in ["train", "val"], "split must be either 'train' or 'val'"
        self.split = split
        self.mixed_tree_ds = mixed_tree_ds
        self.node_list = mixed_tree_ds.get_node_list(split=split)
        
        print(f"Size of {split} dataset:", len(self.node_list))
        self.pad_token_id = pad_token_id
        shuffle(self.node_list)
        self.teacher_models = teacher_models
        for teacher_model in self.teacher_models:
            teacher_model.eval()

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        node = self.node_list[idx]
        assert node.root.name.startswith("root"), "Roots coming from MixedTreeDataStructure should be named 'root<id>'"
        root_id = int(node.root.name[4:])
        tokens_tensor = self.mixed_tree_ds.get_tokens_by_root_id(root_id, self.split)
        leaf_tokens_indx_path = node.get_tokens_indx_path()
        sequence_list = [tokens_tensor[indx] for indx in leaf_tokens_indx_path]
        sequence_list_unpadded = [seq[seq != self.pad_token_id].unsqueeze(0) for seq in sequence_list]
        num_prompt_tokens = sequence_list_unpadded[0].shape[1]
        num_non_prompt_tokens = sum([seq.shape[1] for seq in sequence_list_unpadded]) - num_prompt_tokens
        node_sequence = torch.cat(sequence_list_unpadded, dim=1)
        if node_sequence.dtype != torch.int64:
            node_sequence = node_sequence.to(torch.int64)
        node_sequence = node_sequence.to(self.teacher_models[0].device)
        # Get the targets for every token from the teacher models
        with torch.no_grad():
            # Get the value predictions from each teacher model and average them
            value_predictions = []
            for teacher_model in self.teacher_models:
                # Each teacher model should return a value prediction for every non-prompt token, meaning every token which does not
                # belong to the root node.
                outputs = teacher_model(input_ids=node_sequence.squeeze().unsqueeze(0), 
                                        attention_mask=torch.ones_like(node_sequence).squeeze().unsqueeze(0),
                                        num_logits_to_return=num_non_prompt_tokens)
                logits = outputs.logits.squeeze().cpu()
                if logits.numel() == 1:
                    logits = logits.unsqueeze(0)
                value_predictions.append(logits)
            values = torch.stack(value_predictions, dim=0).squeeze()
            if values.dim() == 1:
                values = values.unsqueeze(1)
            values = values.transpose(0,1)  # [num_non_prompt_tokens, num_objectives]

        return {"input_ids": node_sequence.squeeze(), "attention_mask": torch.ones_like(node_sequence.squeeze()), "labels": values, "first_non_prompt_indx": num_prompt_tokens}

# NOTE: The order in which the teacher models are listed determines which value head in the student model corresponds to which objective
class TreeDataset_Distill(TreeDataset_HDF5):
    def __init__(self, root_dict, tokens_file, pad_token_id=32000, teacher_models=[]):
        assert isinstance(tokens_file, str) and tokens_file.endswith(".hdf5"), "tokens_file must be a path to an HDF5 file"
        assert len(teacher_models) > 0, "At least one teacher model must be provided"
        self.tokens_file = File(tokens_file, "r")
        self.node_list = []
        # Create a list of all leaf nodes from all trees
        for prompt_name, root in root_dict.items():
            root.name = prompt_name
            assert type(root) is ValNodeTokensHDF5
            cur_node_list = [node for node in at.PreOrderIter(root) if node.is_leaf]
            self.node_list = self.node_list + cur_node_list
        # print("Size of dataset:", len(self.node_list))
        self.pad_token_id = pad_token_id
        shuffle(self.node_list)
        self.teacher_models = teacher_models
        for teacher_model in self.teacher_models:
            teacher_model.eval()

    def __len__(self):
        return len(self.node_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        node = self.node_list[idx]
        dataset_name = node.root.name
        leaf_tokens_indx_path = node.get_tokens_indx_path()
        sequence_list = [torch.Tensor(self.tokens_file[dataset_name][indx]) for indx in leaf_tokens_indx_path]
        sequence_list_unpadded = [seq[seq != self.pad_token_id].unsqueeze(0) for seq in sequence_list]
        num_prompt_tokens = sequence_list_unpadded[0].shape[1]
        num_non_prompt_tokens = sum([seq.shape[1] for seq in sequence_list_unpadded]) - num_prompt_tokens
        node_sequence = torch.cat(sequence_list_unpadded, dim=1)
        if node_sequence.dtype != torch.int64:
            node_sequence = node_sequence.to(torch.int64)
        node_sequence = node_sequence.to(self.teacher_models[0].device)
        # Get the targets for every token from the teacher models
        with torch.no_grad():
            # Get the value predictions from each teacher model and average them
            value_predictions = []
            for teacher_model in self.teacher_models:
                # Each teacher model should return a value prediction for every non-prompt token, meaning every token which does not
                # belong to the root node.
                outputs = teacher_model(input_ids=node_sequence.squeeze().unsqueeze(0), 
                                        attention_mask=torch.ones_like(node_sequence).squeeze().unsqueeze(0),
                                        num_logits_to_return=num_non_prompt_tokens)
                logits = outputs.logits.squeeze().cpu()
                if logits.numel() == 1:
                    logits = logits.unsqueeze(0)
                value_predictions.append(logits)
            values = torch.stack(value_predictions, dim=0).squeeze()
            if values.dim() == 1:
                values = values.unsqueeze(1)
            values = values.transpose(0,1)  # [num_non_prompt_tokens, num_objectives]
        # We could try to correct the targets based on the rewards from the reward models, but I'm not trying that yet
        # value = node.value[self.objective] if type(node.value) is dict else node.value

        return {"input_ids": node_sequence.squeeze(), "attention_mask": torch.ones_like(node_sequence.squeeze()), "labels": values, "first_non_prompt_indx": num_prompt_tokens}

from transformers import DataCollatorWithPadding
class TokenRegressionCollatorND:
    def __init__(self, tokenizer, num_targets=3, ignore_special=True, pad_value=0.0):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        self.num_targets = num_targets
        self.ignore_special = ignore_special
        self.pad_value = pad_value

    def __call__(self, features):
        labels_list = [f.pop("labels") for f in features]   # each: [Li, D]
        batch = self.base(features)  # pads input_ids, attention_mask, special_tokens_mask, etc.
        B = len(labels_list)
        Lmax = batch["input_ids"].size(1)
        D = self.num_targets

        labels = torch.full((B, Lmax, D), self.pad_value, dtype=torch.float32)
        mask = torch.zeros((B, Lmax), dtype=torch.bool)

        for i, labs in enumerate(labels_list):
            first_non_prompt_indx = features[i]["first_non_prompt_indx"]
            arr = torch.tensor(labs, dtype=torch.float32)  # [Li, D]
            L = min(arr.size(0), Lmax)
            labels[i, :L, :] = arr[:L, :]
            mask[i, :L] = True # The labels actually start at the beginning, and extra padding is added at the end for the prompt tokens

        # optionally ignore special tokens
        if self.ignore_special and "special_tokens_mask" in batch:
            stm = batch["special_tokens_mask"].bool()
            mask &= ~stm

        batch["labels"] = labels            # [B, Lmax, D]
        batch["labels_mask"] = mask         # [B, Lmax]
        return batch

class TokenRegressionTrainerND(Trainer):
    def __init__(self, *args, dimension_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimension_weights = None
        if dimension_weights is not None:
            self.dimension_weights = torch.tensor(dimension_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")              # [B, L, D]
        B, L, D = labels.size()
        labels_mask = inputs.pop("labels_mask")    # [B, L]
        # We do not actually worry about the attention mask
        attention_mask = inputs.get("attention_mask")  # [B, L] or None

        # Get the number of unpadded labels per batch element
        num_labels_per_batch = labels_mask.sum(dim=1)  # [B]
        max_num_labels = num_labels_per_batch.max().item()

        first_non_prompt_indx = inputs.pop("first_non_prompt_indx", None)  # Not used by the model
        # Note that we are having the model return logits for all input tokens so that we can extract the ones which we need to compute
        # the loss over.
        outputs = model(**inputs, num_logits_to_return=inputs["input_ids"].size(1))
        # The teacher models only return logits for the non-prompt tokens, but these labels are then padded so that they can be batched.
        # Specifically, they are padded to the max sequence length including both prompt and non-prompt tokens.
        # When the labels are batched together, all the padding is placed on the right, even though the actual starting index of the 
        # non-prompt tokens may not be aligned. Therefore, we need to get logits from the student model for all tokens, then extract the
        # ones which correspond to non-prompt tokens
        # For the ith element in the batch, the range of tokens we can get logits for starts at first_non_prompt_indx[i] and goes to the end
        # of the sequence. We will only take the first max_num_labels logits however, since anything beyond that must be padding.
        device = outputs.logits.device
        logits = outputs.logits
        base = torch.arange(max_num_labels, device=device).unsqueeze(0).expand(B, -1)
        idx = first_non_prompt_indx.unsqueeze(1) + base
        # This mask seems to be redundant since labels_mask stores the same thing, and we overwrite it with that later on
        mask = base < num_labels_per_batch.unsqueeze(1)   # [B, max_num_labels]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(idx)
        max_idx = max_num_labels + first_non_prompt_indx.max().item()
        if max_idx >= logits.shape[1]:
            logits = torch.nn.functional.pad(logits, (0,0,0,max_idx-logits.shape[1]), value=0.0)
        logits = logits[batch_idx, idx]  # [B, max_num_labels, D]

        # Now remove the padding that is shared across all batch elements of the labels
        labels = labels[:, labels_mask.any(dim=0), :]
        labels_mask = labels_mask[:, labels_mask.any(dim=0)]

        # squared error per dim
        # sq_err = (logits[:,:,0] - labels[:,:,0]).unsqueeze(2) ** 2            # [B, L, D]
        sq_err = (logits - labels) ** 2            # [B, L, D]
        if self.dimension_weights is not None:
            w = self.dimension_weights.to(logits.device).view(1, 1, -1)
            sq_err = sq_err * w

        per_token = sq_err.mean(dim=-1)            # [B, L] (average over D)
        mask = labels_mask
        # if attention_mask is not None:
        #     mask = mask & attention_mask.bool()

        eps = 1e-8
        loss = (per_token * mask).sum() / (mask.sum().clamp_min(1) + eps)
        return (loss, outputs) if return_outputs else loss
        return loss
    
def freeze_llama_backbone(m):
    # Freeze LlamaModel backbone
    for p in m.model.parameters():
        p.requires_grad = False

    # Ensure the head stays trainable
    for p in m.score.parameters():
        p.requires_grad = True

    # Quick sanity check
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    print("Trainable params:", trainable[:10], "..." if len(trainable) > 10 else "")