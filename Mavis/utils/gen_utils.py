import torch
import numpy as np
from time import time

# Since we use top-k sampling as explained in Appendix F, we need to get the top-k probabilities and the corresponding token indices
def get_top_k_logits(logits, k, temperature):
    logits = logits/temperature
    last_token_logits = logits[:, -1, :]
    # last_token_probs = torch.softmax(last_token_logits, dim=-1)
    return torch.topk(last_token_logits, k=k, dim=-1)

# This function performs the tilting of the reference model's distribution, as in the MAVIS decoding algorithm
# However, it is done in the logit space for numerical stability
def get_combined_probs(top_k_logits, beta, values):
    if values.shape[0] != top_k_logits.shape[0] or values.shape[1] != top_k_logits.shape[1]:
        values = values.transpose(0, 1)  # Should be (batch_size, top_k)
    modified_logits = top_k_logits + (beta * values)
    return torch.softmax(modified_logits, dim=-1)

# Use this to get a function that can either be passed to the eval_loop function in search_utils.py or used directly
# When given a prompt, the function obtained from get_mavis_alg will apply MAVIS decoding to generate a sequence
# The function will either return the full tokenized sequence or the decoded string, depending on the return_strings argument
# If track_KL is set to True, the function will also return the log probability ratio
def get_mavis_alg(tokenizer, generative_model, value_generator, max_new_tokens, args, device="cuda", return_strings=False, 
                  verbose=False, return_log_prob_ratio=False, track_KL=False, num_parallel=1, force_no_eos_on_first_token=False):
    # assert not (type(value_generator) is ValueGeneratorBatched and track_KL), "Cannot do batched generation while tracking the KL divergence"
    eos_token_id = tokenizer.eos_token_id
    device = args.device if "device" in args else device
    if track_KL:
        # We need this variable to stay in scope inbetween calls to mavis
        sequence_kl_div_list = []
    def mavis(input, max_new_tokens=max_new_tokens, allow_eos_on_first_token=(not force_no_eos_on_first_token)):
        # input can either be a single string or a tensor of input_ids; to generate from multiple different prompts, you must 
        # tokenize them first
        if type(input) is str:
            input_ids = tokenizer(input, return_tensors="pt")["input_ids"].to(device)
        else:
            input_ids = input.to(device)
        if num_parallel > 1:
            assert input_ids.shape[0] == 1, "When using num_parallel > 1, input should not already be batched."
            input_ids = input_ids.repeat((num_parallel, 1))
        prompt_shape = input_ids.shape
        batched = prompt_shape[0] > 1
        if batched: 
            assert isinstance(
                value_generator, (ValueGeneratorBatched, ValueGeneratorMultiHeadBatched)
            ), "A batched value generator (ValueGeneratorBatched or ValueGeneratorMultiHeadBatched) must be used for batched input"
            # Tensor to store the finished sequences
            finished_seq = tokenizer.pad_token_id * torch.ones((prompt_shape[0], prompt_shape[1] + max_new_tokens), dtype=torch.bool, device=args.device)
            finished_seq_indx = None
            seq_pos_indx = torch.arange(prompt_shape[0], device=args.device).unsqueeze(1)
            probs = torch.ones((prompt_shape[0], 1), dtype=torch.float32, device=args.device)  # Probabilities for each sequence
            finished_probs = -1*torch.ones((prompt_shape[0], 1), dtype=torch.float32, device=args.device)  # Probabilities for finished sequences
        tokens_generated = 0
        gen_past_kvs = None

        if track_KL or return_log_prob_ratio:
            log_prob_ratio_list = []
        
        if not args.no_cache: value_generator.reset() # Empty the value models' caches from previous prompts

        return_stats = {}
        generation_start_time = time()

        # Generate N samples
        with torch.no_grad():
            while(tokens_generated < max_new_tokens):
                if gen_past_kvs is None or args.no_cache:
                    model_outputs = generative_model(input_ids=input_ids, use_cache=(not args.no_cache), return_dict=True)
                else:
                    # For Llama models, it is required to give cache_position as an argument if you are passing in a cache (via past_key_values)
                    # cache_position should be the index for the new token which was not present in the last forward pass; this is simply 
                    # y.shape[1]-1 for the unbatched case, but we need to provide it as a tensor
                    model_inputs = generative_model.prepare_inputs_for_generation(input_ids, past_key_values=gen_past_kvs, cache_position=torch.tensor(input_ids.shape[1]-1).unsqueeze(0).to(device), use_cache=True)
                    model_outputs = generative_model(**model_inputs, return_dict=True)
                logits = model_outputs.logits
                if not allow_eos_on_first_token and tokens_generated == 0:
                    logits[:, -1, eos_token_id] = -1e10  # Set the probability of generating EOS to a very small value
                if not args.no_cache:
                    gen_past_kvs = model_outputs.past_key_values

                top_k_logits, top_k_indices = get_top_k_logits(logits, args.topk, args.temperature)
                
                if (type(value_generator) is ValueGeneratorWithCache or type(value_generator) is ValueGeneratorMultiHead):
                    candidate_sequences = torch.cat([input_ids.repeat((top_k_indices.shape[1], 1)), top_k_indices.transpose(0, 1)], dim=1)
                    values = value_generator.get_values(candidate_sequences).to(device)
                elif isinstance(value_generator, (ValueGeneratorBatched, ValueGeneratorMultiHeadBatched)):
                    values = value_generator.get_values(input_ids, top_k_indices).to(device)
                else:
                    # Default to the same behavior as ValueGeneratorWithCache
                    candidate_sequences = torch.cat([input_ids.repeat((top_k_indices.shape[1], 1)), top_k_indices.transpose(0, 1)], dim=1)
                    values = value_generator.get_values(candidate_sequences).to(device)

                if len(values.shape) == 3:
                    values = values.squeeze(2)  # Remove the extra dimension if present

                # values will be reshaped to match top_k_probs if needed here
                combined_probs = get_combined_probs(top_k_logits, args.beta, values)
                
                # Sample from the combined probabilities
                sampled_token_idx = torch.multinomial(combined_probs, num_samples=1).squeeze()
                sampled_token = top_k_indices[torch.arange(top_k_indices.shape[0]), sampled_token_idx]  # Now a scalar tensor
                sampled_token = sampled_token.unsqueeze(1)

                if batched:
                    finished_seq_indx = (sampled_token == eos_token_id).squeeze()
                
                if track_KL or return_log_prob_ratio:
                    top_k_probs = torch.topk(torch.softmax(logits[:, -1, :]/args.temperature, dim=-1), k=args.topk, dim=-1).values
                    ref_prob = top_k_probs[torch.arange(top_k_probs.shape[0]), sampled_token_idx]
                    mavis_prob = combined_probs[torch.arange(combined_probs.shape[0]), sampled_token_idx]
                    log_prob_ratio = torch.log(mavis_prob/ref_prob)
                    if batched:
                        # If a sequence finishes early, use a zero for the log prob ratio. The tensor seq_pos_indx
                        # stores the indices of the sequences which are not finished yet, and those are the indices
                        # which need to be filled
                        full_log_prob_ratio = torch.zeros(prompt_shape[0], device=device, dtype=log_prob_ratio.dtype)
                        full_log_prob_ratio[seq_pos_indx.squeeze()] = log_prob_ratio
                        log_prob_ratio = full_log_prob_ratio
                    else:
                        log_prob_ratio = log_prob_ratio.unsqueeze(0)
                    log_prob_ratio_list.append(log_prob_ratio)

                # Now concatenate with matching dimensions
                input_ids = torch.cat([input_ids, sampled_token], dim=1)  # Both are 2D tensors
                
                tokens_generated += 1
                if(verbose and tokens_generated %20==0):
                    print(f"Tokens generated: {tokens_generated}")

                if batched:
                    probs = probs * combined_probs[torch.arange(combined_probs.shape[0]), sampled_token_idx].unsqueeze(1)
                    # Use seq_pos_indx to place the finished sequences in the correct position in the finished_seq tensor
                    if finished_seq_indx.any():
                        finished_seq[seq_pos_indx[finished_seq_indx], :prompt_shape[1]+tokens_generated] = input_ids[finished_seq_indx].unsqueeze(1)
                        input_ids = input_ids[~finished_seq_indx]
                        finished_probs[seq_pos_indx[finished_seq_indx].squeeze()] = probs[finished_seq_indx]
                        probs = probs[~finished_seq_indx]
                        seq_pos_indx = seq_pos_indx[~finished_seq_indx]
                    if input_ids.shape[0] == 0:
                        break
                    # Remove finished sequences from the cache
                    if gen_past_kvs is not None and finished_seq_indx.any():
                        gen_past_kvs.batch_select_indices(~finished_seq_indx)
                        value_generator.remove_from_cache(finished_seq_indx)
                else:
                    if sampled_token.item() == eos_token_id:
                        break
        generation_time_elapsed = time() - generation_start_time
        return_stats["tokens_per_second"] = tokens_generated / generation_time_elapsed
        if verbose:
            print(f"Average tokens per second: {tokens_generated / generation_time_elapsed:.2f}")
        if track_KL:
            log_prob_ratios_stacked = torch.stack(log_prob_ratio_list, dim=1)  # Stack along the sequence dimension
            avg_log_prob_ratio = torch.sum(log_prob_ratios_stacked, dim=1)
            return_stats["sequence_KL"] = avg_log_prob_ratio.cpu().numpy().tolist() # This gets used by eval_loop for reporting KL divergence
            sequence_kl_div_list.append(avg_log_prob_ratio.mean().item()) # This is only used for printing running average
            if verbose:
                print(f"Running average log probability ratio: {torch.mean(sequence_kl_div_list):.4f}")
        if batched:
            if input_ids.shape[0] > 0:
                finished_seq[seq_pos_indx.squeeze(), :input_ids.shape[1]+tokens_generated] = input_ids
                finished_probs[seq_pos_indx.squeeze()] = probs
        else:
            finished_seq = input_ids
            # finished_probs = probs
        if return_log_prob_ratio:
            # This is not used with search_utils.eval_loop, so no need to return stats
            log_prob_ratios_stacked = torch.stack(log_prob_ratio_list, dim=1)  # Stack along the sequence dimension
            return finished_seq, log_prob_ratios_stacked
        if return_strings:
            return tokenizer.batch_decode(finished_seq, skip_special_tokens=True), return_stats
        else:
            return finished_seq, return_stats
    return mavis

class ValueGeneratorWithCache():
    def __init__(self, value_models, val_tokenizer, device, obj_weights, dtype=torch.float16):
        self.value_models = value_models
        self.val_tokenizer = val_tokenizer
        self.device = device
        self.dtype = dtype
        self.obj_weights = obj_weights
        # We assume that the objective weights have already been checked for validity, and that only those with weights > 0 
        # are present in value_models
        self.objective_names = list(self.value_models.keys())
        self.val_past_kvs = {k: None for k in value_models.keys()}
        
    def get_values(self, input_ids):
        values = {}
        for k, v in self.value_models.items():
            with torch.no_grad():
                if self.val_past_kvs[k] is None:
                    outputs = v(input_ids, return_dict=True, use_cache=True)
                else:
                    # For future reference: If it is possible to reuse the KVs cached for the chosen candidate, then we can replace
                    # the -2 with a -1 so that we are not recomputing the KVs for the token that was chosen on the
                    # previous step
                    new_tokens = input_ids[:, -2:] if self.val_past_kvs[k] is not None else input_ids
                    outputs = v(new_tokens, past_key_values=self.val_past_kvs[k], 
                                        return_dict=True, use_cache=True)
                # Store values and update cache
                values[k] = outputs.logits.cpu().detach()
                if hasattr(outputs, "past_key_values"):
                    self.val_past_kvs[k] = outputs.past_key_values
        self.truncate_cache(1)  # Remove the last token from the cache
        # Compute combined output
        assert list(values.keys()) == list(self.value_models.keys()), f"Expected keys {list(self.value_models.keys())}, but got {list(values.keys())}"
        outputs = torch.sum(torch.stack([values[k] * self.obj_weights[k] for k in self.objective_names]), dim=0)
        return outputs
        
    def reset(self):
        """Reset the KV cache between different prompts"""
        self.val_past_kvs = {k: None for k in self.objective_names}

    def truncate_cache(self, num_to_remove):
        """Remove data from the KV cache corresponding to the last num_to_remove tokens
            This is needed because we evaluate multiple candidate tokens but only keep one for the next step"""
        for k in self.val_past_kvs.keys():
            prev_len = self.val_past_kvs[k].get_seq_length()
            if self.val_past_kvs[k] is not None:
                # Assuming the cache is a tuple of (key, value)
                self.val_past_kvs[k].key_cache = [x[:, :, :-num_to_remove, :] for x in self.val_past_kvs[k].key_cache]
                self.val_past_kvs[k].value_cache = [x[:, :, :-num_to_remove, :] for x in self.val_past_kvs[k].value_cache]
            assert self.val_past_kvs[k].get_seq_length() == prev_len - num_to_remove, f"Cache length not updated correctly for {k}. Expected {prev_len - num_to_remove}, got {self.val_past_kvs[k].get_seq_length()}"
    
    def set_obj_weights(self, obj_weights):
        weights_vec = torch.tensor([obj_weights[k] for k in self.objective_names], device=self.device)
        assert torch.all(weights_vec >= 0), "Objective weights must be non-negative"
        assert torch.all(weights_vec <= 1), "Objective weights must be less than or equal to 1"
        assert torch.isclose(weights_vec.sum().to(torch.float64), torch.tensor(1.0, device=self.device, dtype=torch.float64)), "Objective weights must sum to 1"
        self.obj_weights = obj_weights

class ValueGeneratorMultiHead():
    def __init__(self, value_model, output_name_map, val_tokenizer, device, obj_weights, dtype=torch.float16):
        self.value_model = value_model
        # This allows us to map the indices in the output of the value model to the correct objective names
        self.output_name_map = output_name_map
        self.val_tokenizer = val_tokenizer
        self.device = device
        self.dtype = dtype
        self.obj_weights = obj_weights
        # We assume that the objective weights have already been checked for validity, and that only those with weights > 0 
        # are present in value_models
        self.objective_names = list(self.obj_weights.keys())
        self.val_past_kvs = None
    
    def get_values(self, input_ids):
        values = {}
        with torch.no_grad():
            if self.val_past_kvs is None:
                outputs = self.value_model(input_ids, return_dict=True, use_cache=True)
            else:
                # For future reference: If it is possible to reuse the KVs cached for the chosen candidate, then we can replace
                # the -2 with a -1 so that we are not recomputing the KVs for the token that was chosen on the
                # previous step
                new_tokens = input_ids[:, -2:] if self.val_past_kvs is not None else input_ids
                outputs = self.value_model(new_tokens, past_key_values=self.val_past_kvs,
                                            return_dict=True, use_cache=True)
            # Store values and update cache
            values = outputs.logits.cpu().detach()
            if hasattr(outputs, "past_key_values"):
                self.val_past_kvs = outputs.past_key_values
        self.truncate_cache(1)  # Remove the last token from the cache
        # Compute combined output
        # TODO: make this more efficient using a matrix-vector product
        per_obj_values = [values[:,i] * self.obj_weights[self.output_name_map[i]] for i in range(values.shape[1]) if self.obj_weights[self.output_name_map[i]] > 0.0]
        outputs = torch.sum(torch.stack(per_obj_values), dim=0)
        if len(outputs.shape) < 2:
            outputs = outputs.unsqueeze(1)  # Make sure output size is correct
        return outputs
    
    def reset(self):
        """Reset the KV cache between different prompts"""
        self.val_past_kvs = None

    def truncate_cache(self, num_to_remove):
        """Remove data from the KV cache corresponding to the last num_to_remove tokens
            This is needed because we evaluate multiple candidate tokens but only keep one for the next step"""
        prev_len = self.val_past_kvs.get_seq_length()
        if self.val_past_kvs is not None:
            # Assuming the cache is a tuple of (key, value)
            self.val_past_kvs.key_cache = [x[:, :, :-num_to_remove, :] for x in self.val_past_kvs.key_cache]
            self.val_past_kvs.value_cache = [x[:, :, :-num_to_remove, :] for x in self.val_past_kvs.value_cache]
        assert self.val_past_kvs.get_seq_length() == prev_len - num_to_remove, f"Cache length not updated correctly. Expected {prev_len - num_to_remove}, got {self.val_past_kvs.get_seq_length()}"

    def set_obj_weights(self, obj_weights):
        weights_vec = torch.tensor([obj_weights[k] for k in self.objective_names], device=self.device)
        assert torch.all(weights_vec >= 0), "Objective weights must be non-negative"
        assert torch.all(weights_vec <= 1), "Objective weights must be less than or equal to 1"
        assert torch.isclose(weights_vec.sum().to(torch.float64), torch.tensor(1.0, device=self.device, dtype=torch.float64)), "Objective weights must sum to 1"
        self.obj_weights = obj_weights

# This class uses the method from Q# to get value for all the candidates for a given sequence in a single forward pass.
# NOTE: From my observations, doing a batched forward pass with the value model gives slightly different logits than doing
# a single forward pass when the model is run in half precision
class ValueGeneratorBatched(ValueGeneratorWithCache):
    def __init__(self, value_models, val_tokenizer, device, obj_weights, dtype=torch.float16):
        super().__init__(value_models, val_tokenizer, device, obj_weights, dtype=dtype)
        # Need to handle this differently depending on whether we are using a value model with a 
        # peft adapter
        if hasattr(value_models[list(value_models.keys())[0]].model, "model"):
            self.prep_causal_mask_fn = value_models[list(value_models.keys())[0]].model.model._prepare_4d_causal_attention_mask_with_cache_position
        else:
            self.prep_causal_mask_fn = value_models[list(value_models.keys())[0]].model._prepare_4d_causal_attention_mask_with_cache_position
        
        # Cache frequently used tensors to avoid recomputation
        self._diagonal_mask_cache = {}      # Cache by (top_k, dtype_str, device_str)
        self._position_ids_cache = {}       # Cache by (batch_size, num_new_tokens, num_past_tokens)
        self._cache_position_cache = {}     # Cache by (full_seq_len, num_new_tokens)

    def _get_diagonal_mask(self, top_k, dtype, device):
        """Get or create diagonal mask for candidate tokens that prevents them from attending to each other"""
        # Use string representations for hashing since tensors aren't hashable
        cache_key = (top_k, str(dtype), str(device))
        if cache_key not in self._diagonal_mask_cache:
            diagonal_mask = torch.full((top_k, top_k), torch.finfo(dtype).min, dtype=dtype, device=device)
            diagonal_mask.fill_diagonal_(0)
            self._diagonal_mask_cache[cache_key] = diagonal_mask
        return self._diagonal_mask_cache[cache_key]
    
    def _get_position_ids(self, batch_size, num_new_tokens, num_past_tokens):
        """Get or create position IDs tensor for the new tokens"""
        cache_key = (batch_size, num_new_tokens, num_past_tokens)
        if cache_key not in self._position_ids_cache:
            position_ids = num_past_tokens * torch.ones((batch_size, num_new_tokens), 
                                                       device=self.device, dtype=torch.long)
            self._position_ids_cache[cache_key] = position_ids
        return self._position_ids_cache[cache_key].clone()  # Clone to avoid modifying cached version
    
    def _get_cache_position(self, full_seq_len, num_new_tokens):
        """Get or create cache position array for attention computation"""
        cache_key = (full_seq_len, num_new_tokens)
        if cache_key not in self._cache_position_cache:
            cache_pos = torch.arange(full_seq_len - num_new_tokens, full_seq_len, device=self.device)
            self._cache_position_cache[cache_key] = cache_pos
        return self._cache_position_cache[cache_key]

    def reset(self):
        """Reset the KV cache between different prompts and clear attention caches"""
        super().reset()  # Call parent reset method
        self.clear_attention_caches()  # Clear our attention mask caches

    def get_values(self, past_input_ids, top_k_indices):
        num_past_tokens = past_input_ids.shape[1]
        batch_size = past_input_ids.shape[0]
        top_k = top_k_indices.shape[1]
        # Concatenate the past input IDs with the top-k indices to form the full input IDs
        input_ids = torch.cat([past_input_ids, top_k_indices], dim=1)
        full_seq_len = input_ids.shape[1]
        if self.cache_is_empty():
            # First get the keys and values for past_input_ids (which will be everything except for the candidates)
            for k, v in self.value_models.items():
                with torch.no_grad():
                    outputs = v(past_input_ids, return_dict=True, use_cache=True)
                if hasattr(outputs, "past_key_values"):
                    self.val_past_kvs[k] = outputs.past_key_values
            num_new_tokens = top_k
            all_non_candidate_tokens_in_cache = True
        else:
            num_new_tokens = top_k + 1  # Only the last non-candidate token and all the candidate tokens are new
            all_non_candidate_tokens_in_cache = False
        values = {}
        for k, v in self.value_models.items():
            with torch.no_grad():
                # Use cached cache_position array
                cache_position = self._get_cache_position(full_seq_len, num_new_tokens)
                
                attn_mask_4d = self.prep_causal_mask_fn(
                    attention_mask=torch.ones((batch_size, full_seq_len), device=self.device, dtype=self.dtype),
                    sequence_length=num_new_tokens, 
                    target_length=full_seq_len,
                    dtype=self.dtype,
                    device=self.device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                )
                
                # Use cached diagonal mask for candidate tokens
                diagonal_mask = self._get_diagonal_mask(top_k, attn_mask_4d.dtype, self.device)
                attn_mask_4d[:, :, -top_k:, -top_k:] = diagonal_mask
                
                # Use cached position IDs
                position_ids = self._get_position_ids(batch_size, num_new_tokens, num_past_tokens)
                if not all_non_candidate_tokens_in_cache:
                    # Prevent the last non-candidate token from attending to the candidate tokens
                    attn_mask_4d[:, :, -num_new_tokens, -top_k:] = torch.finfo(attn_mask_4d.dtype).min
                    # Need to properly set the position id for the last non-candidate token
                    position_ids[:,0] = num_past_tokens-1
                outputs = v(input_ids[:, -num_new_tokens:], attention_mask=attn_mask_4d, position_ids=position_ids,
                            past_key_values=self.val_past_kvs[k], return_dict=True, use_cache=True,
                            num_logits_to_return=top_k)
                # Store values and update cache
                values[k] = outputs.logits.cpu().detach().squeeze(0)
                if hasattr(outputs, "past_key_values"):
                    self.val_past_kvs[k] = outputs.past_key_values
        self.truncate_cache(top_k)  # Remove the part of the the cache corresponding to the candidate tokens
        # Compute combined output
        assert list(values.keys()) == list(self.value_models.keys()), f"Expected keys {list(self.value_models.keys())}, but got {list(values.keys())}"
        outputs = torch.sum(torch.stack([values[k] * self.obj_weights[k] for k in self.objective_names]), dim=0)
        return outputs
    
    def cache_is_empty(self):
        """Check if the KV cache is empty"""
        return all(v is None for v in self.val_past_kvs.values())
    
    # This function should be used when the batch size decreases due to some sequences reaching an EOS token before others.
    def remove_from_cache(self, indices_to_remove):
        """Remove the specified indices from the KV cache"""
        for k in self.val_past_kvs.keys():
            if self.val_past_kvs[k] is not None:
                self.val_past_kvs[k].batch_select_indices(~indices_to_remove)
    
    # This function should be used during the first iteration of beam search when the batch size is expanded
    # It creates copies of the KV cache to accomodate the new batch size
    def expand_beams(self, num_beams):
        """Expand the KV cache to accomodate a new batch size"""
        for k in self.val_past_kvs.keys():
            if self.val_past_kvs[k] is not None:
                self.val_past_kvs[k].batch_repeat_interleave(num_beams)

    # This function should be used during successive iterations of beam search when beams are replaced
    def update_beams(self, new_beam_indices):
        """Update the KV cache to only keep the beams specified by new_beam_indices"""
        for k in self.val_past_kvs.keys():
            if self.val_past_kvs[k] is not None:
                old_kvs_split = self.val_past_kvs[k].batch_split(len(new_beam_indices), 1)
                new_kvs_split = [old_kvs_split[i] for i in new_beam_indices]
                self.val_past_kvs[k].from_batch_splits(new_kvs_split)

    def clear_attention_caches(self):
        """Clear attention mask caches to free memory when needed"""
        self._diagonal_mask_cache.clear()
        self._position_ids_cache.clear()
        self._cache_position_cache.clear()
    
    def get_cache_stats(self):
        """Get statistics about cache usage for debugging/monitoring"""
        return {
            'diagonal_mask_cache_size': len(self._diagonal_mask_cache),
            'position_ids_cache_size': len(self._position_ids_cache),
            'cache_position_cache_size': len(self._cache_position_cache),
        }
    
class ValueGeneratorMultiHeadBatched(ValueGeneratorMultiHead):
    def __init__(self, value_model, output_name_map, val_tokenizer, device, obj_weights, dtype=torch.float16):
        super().__init__(value_model, output_name_map, val_tokenizer, device, obj_weights, dtype=dtype)
        assert hasattr(value_model.model, "model"), "Value model must be a PEFT model"
        self.prep_causal_mask_fn = value_model.model.model._prepare_4d_causal_attention_mask_with_cache_position
        # Cache frequently used tensors to avoid recomputation
        self._diagonal_mask_cache = {}      # Cache by (top_k, dtype_str, device_str)
        self._position_ids_cache = {}       # Cache by (batch_size, num_new_tokens, num_past_tokens)
        self._cache_position_cache = {}     # Cache by (full_seq_len, num_new_tokens)
    
    def get_values(self, past_input_ids, top_k_indices):
        num_past_tokens = past_input_ids.shape[1]
        batch_size = past_input_ids.shape[0]
        top_k = top_k_indices.shape[1]
        # Concatenate the past input IDs with the top-k indices to form the full input IDs
        input_ids = torch.cat([past_input_ids, top_k_indices], dim=1)
        full_seq_len = input_ids.shape[1]
        if self.cache_is_empty():
            # First get the keys and values for past_input_ids (which will be everything except for the candidates)
            with torch.no_grad():
                outputs = self.value_model(past_input_ids, return_dict=True, use_cache=True)
            if hasattr(outputs, "past_key_values"):
                self.val_past_kvs = outputs.past_key_values
            num_new_tokens = top_k
            all_non_candidate_tokens_in_cache = True
        else:
            num_new_tokens = top_k + 1  # Only the last non-candidate token and all the candidate tokens are new
            all_non_candidate_tokens_in_cache = False
        values = {}
        with torch.no_grad():
            # Use cached cache_position array
            cache_position = self._get_cache_position(full_seq_len, num_new_tokens)
            
            attn_mask_4d = self.prep_causal_mask_fn(
                attention_mask=torch.ones((batch_size, full_seq_len), device=self.device, dtype=self.dtype),
                sequence_length=num_new_tokens, 
                target_length=full_seq_len,
                dtype=self.dtype,
                device=self.device,
                cache_position=cache_position,
                batch_size=batch_size,
            )
            
            # Use cached diagonal mask for candidate tokens
            diagonal_mask = self._get_diagonal_mask(top_k, attn_mask_4d.dtype, self.device)
            attn_mask_4d[:, :, -top_k:, -top_k:] = diagonal_mask
            
            # Use cached position IDs
            position_ids = self._get_position_ids(batch_size, num_new_tokens, num_past_tokens)
            if not all_non_candidate_tokens_in_cache:
                # Prevent the last non-candidate token from attending to the candidate tokens
                attn_mask_4d[:, :, -num_new_tokens, -top_k:] = torch.finfo(attn_mask_4d.dtype).min
                # Need to properly set the position id for the last non-candidate token
                position_ids[:,0] = num_past_tokens-1
            outputs = self.value_model(input_ids[:, -num_new_tokens:], attention_mask=attn_mask_4d, position_ids=position_ids,
                        past_key_values=self.val_past_kvs, return_dict=True, use_cache=True,
                        num_logits_to_return=top_k)
            # Store values and update cache
            values = outputs.logits.cpu().detach()
            if hasattr(outputs, "past_key_values"):
                self.val_past_kvs = outputs.past_key_values
        self.truncate_cache(top_k)  # Remove the part of the the cache corresponding to the candidate tokens
        per_obj_values = []
        for i in range(values.shape[2]):
            if self.output_name_map[i] in self.obj_weights and self.obj_weights[self.output_name_map[i]] > 0.0:
                per_obj_values.append(values[:,:,i] * self.obj_weights[self.output_name_map[i]])
        
        outputs = torch.sum(torch.stack(per_obj_values), dim=0)
        if len(outputs.shape) < 2:
            outputs = outputs.unsqueeze(1)  # Make sure output size is correct
        return outputs
    
    def _get_diagonal_mask(self, top_k, dtype, device):
        """Get or create diagonal mask for candidate tokens that prevents them from attending to each other"""
        # Use string representations for hashing since tensors aren't hashable
        cache_key = (top_k, str(dtype), str(device))
        if cache_key not in self._diagonal_mask_cache:
            diagonal_mask = torch.full((top_k, top_k), torch.finfo(dtype).min, dtype=dtype, device=device)
            diagonal_mask.fill_diagonal_(0)
            self._diagonal_mask_cache[cache_key] = diagonal_mask
        return self._diagonal_mask_cache[cache_key]
    
    def _get_position_ids(self, batch_size, num_new_tokens, num_past_tokens):
        """Get or create position IDs tensor for the new tokens"""
        cache_key = (batch_size, num_new_tokens, num_past_tokens)
        if cache_key not in self._position_ids_cache:
            position_ids = num_past_tokens * torch.ones((batch_size, num_new_tokens), 
                                                       device=self.device, dtype=torch.long)
            self._position_ids_cache[cache_key] = position_ids
        return self._position_ids_cache[cache_key].clone()  # Clone to avoid modifying cached version

    def _get_cache_position(self, full_seq_len, num_new_tokens):
        """Get or create cache position array for attention computation"""
        cache_key = (full_seq_len, num_new_tokens)
        if cache_key not in self._cache_position_cache:
            cache_pos = torch.arange(full_seq_len - num_new_tokens, full_seq_len, device=self.device)
            self._cache_position_cache[cache_key] = cache_pos
        return self._cache_position_cache[cache_key]
    
    def clear_attention_caches(self):
        """Clear attention mask caches to free memory when needed"""
        self._diagonal_mask_cache.clear()
        self._position_ids_cache.clear()
        self._cache_position_cache.clear()

    def reset(self):
        super().reset()
        self.clear_attention_caches()  # Clear our attention mask caches
    
    def cache_is_empty(self):
        return self.val_past_kvs is None

    def remove_from_cache(self, indices_to_remove):
        """Remove the specified indices from the KV cache"""
        if self.val_past_kvs is not None:
            self.val_past_kvs.batch_select_indices(~indices_to_remove)
    
    def expand_beams(self, num_beams):
        """Expand the KV cache to accomodate a new batch size"""
        if self.val_past_kvs is not None:
            self.val_past_kvs.batch_repeat_interleave(num_beams)
    
    def update_beams(self, new_beam_indices):
        """Update the KV cache to only keep the beams specified by new_beam_indices"""
        if self.val_past_kvs is not None:
            old_kvs_split = self.val_past_kvs.batch_split(len(new_beam_indices), 1)
            new_kvs_split = [old_kvs_split[i] for i in new_beam_indices]
            self.val_past_kvs.from_batch_splits(new_kvs_split)
    
    def get_cache_stats(self):
        """Get statistics about cache usage for debugging/monitoring"""
        return {
            'diagonal_mask_cache_size': len(self._diagonal_mask_cache),
            'position_ids_cache_size': len(self._position_ids_cache),
            'cache_position_cache_size': len(self._cache_position_cache),
        }

# Obsolete class; use ValueGeneratorWithCache or ValueGeneratorBatched instead
class ValueGenerator():
    # We assume everything is on the same device
    # The objective_name argument is only used in the single-objective case
    def __init__(self, value_models, val_tokenizer, gen_tokenizer, device, mix_ratio, objective_name=None):
        assert mix_ratio is not None or objective_name is not None, "Must specify mix ratio or ORM_type"
        self.value_models = value_models
        self.val_tokenizer = val_tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.device = device
        self.mix_ratio = mix_ratio
        self.objective_name = objective_name

    def get_values(self, input_ids):
        values = {}
        for k, v in self.value_models.items():
            with torch.no_grad():
                outputs = v(input_ids)
            values[k] = outputs.logits.cpu().detach()
        if len(values.keys()) == 2:
            outputs = (values["help"] * self.mix_ratio + values["harm"] * (1 - self.mix_ratio))
        elif self.objective_name in values.keys():
            outputs = values[self.objective_name]   
        else:
            raise ValueError("Must specify mix ratio or ORM_type")
        return outputs

from transformers import LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import logging
from typing import Optional, Union, List, Tuple
logger = logging.getLogger(__name__)
from transformers.cache_utils import Cache
# Custom class needed to allow for extracting logits for all of the candidate tokens when ValueGeneratorBatched is used.
class LlamaValueModel(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
    
    # The only modification to the forward pass is how pooled_logits is created; if ValueGeneratorBatched is used then
    # we need to keep the last top_k logits. Note that this change means we cannot properly account for right padding,
    # but we have no reason to use right padding with the value models
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_return: int = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )
        
        # Note that during training, there may be right-padding. But we will never use num_logits_to_return during training
        if num_logits_to_return is not None:
            pooled_logits = logits[:,-num_logits_to_return:,:]
        else:
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            # Transformers is hell-bent on using CausalLMLoss as the loss function even if I change it in the config, so I hard-coded
            # the correct loss function here
            # Note that the squeeze is important!
            loss = torch.nn.MSELoss()(pooled_logits.squeeze(), labels.squeeze())
            # self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

# Generation-related arguments
def add_generation_args(parser, defaults_dict=None):
    default_defaults_dict = {
        "base_model_type": 'llama_7b',
        "dataset": 'anthropic',
        "data_split": "train",
        "device": "cuda:0",
        "temperature": 1.0,
        "topk": 40,
    }
    if defaults_dict is not None:
        default_defaults_dict.update(defaults_dict)
    parser.add_argument("--base_model_type", type=str, choices=['llama_7b', 'llama_13b', 'alpaca'], default=default_defaults_dict["base_model_type"])
    parser.add_argument("--dataset", type=str, choices=['anthropic', 'summary', 'safeRLHF'], default=default_defaults_dict["dataset"])
    parser.add_argument("--data_split", type=str, default=default_defaults_dict["data_split"])
    parser.add_argument("--device", type=str, default=default_defaults_dict["device"])
    parser.add_argument("--temperature", type=float, default=default_defaults_dict["temperature"], help="Temperature for sampling")
    parser.add_argument("--topk", type=int, default=default_defaults_dict["topk"], help="Top-k sampling parameter")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching for the generative model (doesn't work currently)")
    return parser

# Arguments specific to MAVIS
def add_mavis_args(parser, defaults_dict=None):
    default_defaults_dict = {
        "value_model_dir": "Models/value_models/",
        "value_model_iter": "0",
        "beta": 1.0,
        "allow_eos_on_first_token": False,
    }
    if defaults_dict is not None:
        default_defaults_dict.update(defaults_dict)
    parser.add_argument("--value_model_dir", type=str, default=default_defaults_dict["value_model_dir"], help="Directory containing the value models for the objectives; it is assumed that the directory contains a subdirectory for each objective used, with the subdirectories containing the trained value model weights")
    parser.add_argument("--value_model_iter", type=str, default=default_defaults_dict["value_model_iter"], help="The iteration of the value model to use; this is used to load the correct checkpoint from the value_model_dir")
    parser.add_argument("--beta", type=float, default=default_defaults_dict["beta"], help="Parameter to control the strength of value guidance")
    parser.add_argument("--allow_eos_on_first_token", action="store_true", default=default_defaults_dict["allow_eos_on_first_token"], help="By default, the model is prevented from generating an EOS token as the first token of the generated sequence. Setting this flag disables that behavior.")
    return parser