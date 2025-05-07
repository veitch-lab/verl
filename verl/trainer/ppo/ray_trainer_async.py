# verl/trainer/ppo/ray_trainer_async.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications for Asynchronous Overlap based on new verl structure

import os
import uuid
from contextlib import contextmanager
from pprint import pprint
from typing import Any, List, Type, Dict
import torch
from copy import deepcopy
import time
import numpy as np
from omegaconf import OmegaConf, open_dict
from collections import defaultdict # Added import

# Use standard timer if available, otherwise keep codetiming
try:
    from verl.trainer.ppo.ray_trainer import _timer
except ImportError:
    from codetiming import Timer
    @contextmanager
    def _timer(name: str, timing_raw: Dict[str, float]):
        with Timer(name=name, logger=None) as timer:
            yield
        if name not in timing_raw:
            timing_raw[name] = 0
        timing_raw[name] += timer.last


from verl import DataProto
from verl.protocol import DataProtoItem, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics, compute_timing_metrics, reduce_metrics,
    compute_throughout_metrics, process_validation_metrics # Added new imports
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async # Added new imports
from verl.trainer.ppo.ray_trainer import ( # Import necessary components from the NEW standard trainer
    RayPPOTrainer,
    Role,
    ResourcePoolManager,
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    # _timer # Use the standard timer context manager (imported above with try-except)
)
from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn # Assuming standard dataset
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.workers.rollout.async_server import AsyncLLMServerManager # Import async manager

# Keep this helper if needed, it's also in the new standard trainer
def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    # Ensure correct handling if item.batch or item.non_tensor_batch is None
    tensors = item.batch if item.batch is not None else {}
    non_tensors = item.non_tensor_batch if item.non_tensor_batch is not None else {}
    meta_info = item.meta_info if item.meta_info is not None else {}
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=meta_info
    )


class RayPPOAsyncTrainer(RayPPOTrainer):
    """
    Asynchronous PPO Trainer using a combined Actor/Rollout worker group (Hybrid Engine)
    and overlapping rollout/reward computation.
    Inherits from the standard RayPPOTrainer and overrides relevant methods.
    """

    # __init__ is inherited from RayPPOTrainer.
    # Base class __init__ handles worker setup based on role_worker_mapping.

    # Override init_workers to potentially initialize async manager
    def init_workers(self):
        """Init resource pool and worker group, including async manager if needed."""
        # Call base class initialization first. This sets up self.actor_rollout_wg etc.
        super().init_workers()

        # Just sanity-check that the base class did its job
        self.async_rollout_mode = (
            self.config.actor_rollout_ref.rollout.mode == "async"
        )

        if self.async_rollout_mode:
            assert hasattr(self, "async_rollout_manager"), (
                "Base init_workers() was expected to create AsyncLLMServerManager."
            )
            print(
                f"[RayPPOAsyncTrainer] Re-using AsyncLLMServerManager "
                #f"(prefix={self.async_rollout_manager.name_prefix})."
            )


    # Override _validate to use async manager if needed
    def _validate(self):
        """Perform validation, handling async generation via manager if enabled."""
        # This largely follows the new standard trainer's _validate
        # Key difference is how generate_sequences is called in async mode

        reward_tensor_lst = []
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        print("Starting validation...")
        # Use tqdm for progress indication if desired
        # from tqdm import tqdm
        # val_iterator = tqdm(self.val_dataloader, desc="Validation Batches")
        val_iterator = self.val_dataloader # Or just iterate directly

        for i, test_data in enumerate(val_iterator):
            print(f"Processing validation batch {i+1}/{len(self.val_dataloader)}")
            test_batch = DataProto.from_single_dict(test_data)

            # Repeat test batch for multiple validation samples per prompt
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            print(f"  Repeated batch size: {len(test_batch)}")

            # Store original inputs for logging
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # Prepare generation batch (pop non-generation keys)
            # --- This pop logic is copied from the new standard trainer ---
            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                 gen_batch = test_batch.pop(
                     batch_keys=["input_ids", "attention_mask", "position_ids"],
                     non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                 )
            else:
                 gen_batch = test_batch.pop(
                     batch_keys=["input_ids", "attention_mask", "position_ids"],
                     non_tensor_batch_keys=["raw_prompt_ids"] + ["raw_prompt"] if self.async_rollout_mode else [],
                 )
            # --- End copied pop logic ---


            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False, # Validation doesn't need log probs recomputed usually
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'temperature': self.config.actor_rollout_ref.rollout.temperature, # Pass temperature
                'validate': True,
            }
            print(f"  Generation meta info: {gen_batch.meta_info}")


            # Pad batch for world size
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
            print(f"  Padded batch size: {len(gen_batch_padded)}, Pad size: {pad_size}")


            # *** ASYNC/SYNC GENERATION CALL ***
            if not self.async_rollout_mode:
                print("  Generating sequences synchronously...")
                output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
            else:
                print("  Generating sequences asynchronously via manager...")
                # Ensure manager exists if in async mode
                if not hasattr(self, 'async_rollout_manager'):
                    raise RuntimeError("Async mode enabled but async_rollout_manager not initialized.")
                self.async_rollout_manager.wake_up() # Ensure manager/server is active
                output_gen_batch_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)
                self.async_rollout_manager.sleep() # Put manager/server to sleep

            # Unpad results
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
            print(f"  Unpadded output batch size: {len(output_gen_batch)}")


            # Store generated outputs for logging
            output_ids = output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Combine original batch data with generated sequences
            test_batch = test_batch.union(output_gen_batch)

            # Evaluate using the validation reward function
            print("  Evaluating rewards...")
            # Use the reward function provided during init (should handle dict return)
            reward_result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_info = reward_result.get("reward_extra_info", {})

            # Store scores and extra info
            # Sum token-level scores for sequence score, clamp if needed by reward fn
            current_scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(current_scores)
            reward_extra_infos_dict["reward"].extend(current_scores) # Add base reward score
            for key, lst in reward_extra_info.items():
                 reward_extra_infos_dict[key].extend(lst)


            # Store data source if available
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        print("Validation generation and reward calculation finished.")

        # Log sample generations to WandB/Logger (inherited method)
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # Dump generations if configured (inherited method)
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
             self._dump_generations(
                 inputs=sample_inputs,
                 outputs=sample_outputs,
                 scores=sample_scores,
                 reward_extra_infos_dict=reward_extra_infos_dict, # Pass the collected dict
                 dump_path=val_data_dir,
             )


        # Process metrics (using the function from new standard trainer)
        print("Processing validation metrics...")
        data_sources = np.concatenate(data_source_lst, axis=0)
        # Ensure all lists in reward_extra_infos_dict have the same length as sample_scores
        expected_len = len(sample_scores)
        # Filter the dict before passing
        filtered_reward_extra_infos = {k: v for k, v in reward_extra_infos_dict.items() if len(v) == expected_len}
        # Warn about skipped keys
        skipped_keys = [k for k, v in reward_extra_infos_dict.items() if len(v) != expected_len]
        if skipped_keys:
            print(f"Warning: Length mismatch for reward_extra_info keys: {skipped_keys}. Skipping these keys for metric processing.")


        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, filtered_reward_extra_infos)


        # Format metrics for logging (using the logic from new standard trainer)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward" # Determine primary metric
            for var_name, metric2val in var2metric2val.items():
                # Find max N if metrics are like 'mean@N/...'
                n_values = [int(name.split('@')[-1].split('/')[0]) for name in metric2val.keys() if '@' in name]
                n_max = max(n_values) if n_values else 1


                for metric_name, metric_val in metric2val.items():
                    # Categorize metrics for better logging structure
                    is_core_metric = (var_name == core_var) and \
                                     any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and \
                                     (f"@{n_max}" in metric_name or n_max == 1)


                    metric_sec = "val-core" if is_core_metric else "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val


        print(f"Final validation metrics calculated: {metric_dict}")
        return metric_dict


    # Override fit method for async overlap
    def fit(self):
        """
        Training loop with asynchronous overlap between rollout generation
        and reward computation.
        """
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        self._load_checkpoint() # Load checkpoint using inherited method

        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate() # Use overridden _validate
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        self.global_steps += 1
        last_val_metrics = None

        # Use tqdm progress bar like the new trainer
        from tqdm import tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")


        # --- Main Training Loop ---
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # --- Pop generation keys (copied from new trainer) ---
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                     gen_batch = batch.pop(
                         batch_keys=["input_ids", "attention_mask", "position_ids"],
                         non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                     )
                else:
                     gen_batch = batch.pop(
                         batch_keys=["input_ids", "attention_mask", "position_ids"],
                         non_tensor_batch_keys=["raw_prompt_ids"] + ["raw_prompt"] if self.async_rollout_mode else [],
                     )
                # --- End copied pop logic ---


                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # 1. Generate Sequences (Rollout)
                    with _timer('gen', timing_raw):
                        if not self.async_rollout_mode:
                             gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                             # Use the async manager
                             if not hasattr(self, 'async_rollout_manager'):
                                 raise RuntimeError("Async mode enabled but async_rollout_manager not initialized.")
                             self.async_rollout_manager.wake_up()
                             gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                             self.async_rollout_manager.sleep() # Allow server to process if needed

                    # --- REMAX Baseline Generation (if needed) ---
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                         with _timer('gen_max', timing_raw):
                             gen_baseline_batch = deepcopy(gen_batch) # Use the popped gen_batch
                             gen_baseline_batch.meta_info = gen_batch.meta_info.copy()
                             gen_baseline_batch.meta_info['do_sample'] = False
                             if not self.async_rollout_mode:
                                 gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                             else:
                                 self.async_rollout_manager.wake_up()
                                 gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                 self.async_rollout_manager.sleep()

                             temp_baseline_batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                             temp_baseline_batch = temp_baseline_batch.union(gen_baseline_output)
                             reward_baseline_result = self.reward_fn(temp_baseline_batch, return_dict=True)
                             reward_baseline_tensor = reward_baseline_result["reward_tensor"]
                             reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                             batch.batch['reward_baselines'] = reward_baseline_tensor
                             del gen_baseline_batch, gen_baseline_output, temp_baseline_batch
                    # --- End REMAX Handling ---


                    # Add unique IDs for advantage estimators like GRPO/RLOO
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                    # Repeat original batch data to match rollout.n and combine with generated sequences
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)


                    # Add response mask (needed for advantage/KL)
                    batch.batch["response_mask"] = compute_response_mask(batch)


                    # --- Rejection Sampling (Optional) ---
                    # Note: Performing this *before* async reward launch might be better
                    # if the reward used for sampling is slow. Otherwise, the benefit of overlap is reduced.
                    # We calculate a temporary reward here if needed.
                    if self.config.trainer.get("rejection_sample", False):
                         with _timer('rejection_sample_reward', timing_raw):
                             # Calculate reward ONLY for sampling logic below
                             temp_reward_result = self.reward_fn(batch, return_dict=True)
                             sampling_reward_tensor = temp_reward_result["reward_tensor"]

                         with _timer('rejection_sample_logic', timing_raw):
                             uids = batch.non_tensor_batch['uid']
                             unique_uids, uid_indices = np.unique(uids, return_inverse=True)
                             valid_mask = torch.ones(len(uids), dtype=torch.bool, device=sampling_reward_tensor.device)
                             solve_none, solve_all = 0, 0
                             seq_rewards = sampling_reward_tensor.sum(-1)
                             num_prompts = len(unique_uids)

                             for i, uid in enumerate(unique_uids):
                                 prompt_mask = (uid_indices == i)
                                 prompt_rewards = seq_rewards[prompt_mask]
                                 # Adapt thresholds 0.0 and 1.0 if reward scale differs
                                 is_all_zero = (prompt_rewards <= 0.0).all()
                                 is_all_one = (prompt_rewards >= 1.0).all()
                                 if is_all_zero:
                                     valid_mask[prompt_mask] = False; solve_none += 1
                                 elif is_all_one:
                                     valid_mask[prompt_mask] = False; solve_all += 1

                             metrics['batch/solve_none'] = solve_none
                             metrics['batch/solve_all'] = solve_all
                             metrics['batch/solve_partial'] = num_prompts - solve_none - solve_all

                             if not valid_mask.any():
                                 print("Warning: Rejection sampling removed all samples. Skipping step.")
                                 progress_bar.update(1); self.global_steps += 1; continue

                             valid_indices = torch.where(valid_mask)[0]
                             non_valid_indices = torch.where(~valid_mask)[0]
                             num_valid_samples = len(valid_indices)
                             target_total_batch_size = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                             padding_needed = target_total_batch_size - num_valid_samples

                             combined_indices = valid_indices.tolist()
                             if padding_needed > 0 and len(non_valid_indices) > 0:
                                 padding_samples = min(padding_needed, len(non_valid_indices))
                                 perm = torch.randperm(len(non_valid_indices), device=non_valid_indices.device)[:padding_samples]
                                 padding_indices = non_valid_indices[perm]
                                 combined_indices.extend(padding_indices.tolist())

                             final_mask = torch.zeros(len(batch.batch['input_ids']), dtype=torch.bool, device=batch.batch['input_ids'].device)
                             final_mask[torch.tensor(combined_indices, device=final_mask.device)] = True

                             batch = batch[final_mask]
                             # Ensure it's still a DataProto object after slicing
                             if isinstance(batch, DataProtoItem):
                                 batch = dataprotoitem_to_dataproto(batch)

                             metrics['batch/samples_after_sampling'] = len(batch)
                         # Clear the temporary reward tensor
                         del sampling_reward_tensor, temp_reward_result
                    # --- End Rejection Sampling ---


                    # 2. Balance Batch (Optional, inherited from base class)
                    if self.config.trainer.balance_batch:
                         self._balance_batch(batch, metrics=metrics) # Use inherited method


                    # Add global token count (inherited from base class)
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()


                    # 3. Compute Rewards (Potentially Async)
                    reward_extra_infos_dict = {} # Initialize in case reward fn doesn't return it
                    with _timer('reward', timing_raw):
                        if self.use_rm: # Compute RM score if enabled
                             rm_scores = self.rm_wg.compute_rm_score(batch)
                             batch = batch.union(rm_scores)

                        # Launch reward computation (sync or async based on config)
                        if self.config.reward_model.get("launch_reward_fn_async", False):
                            import ray # Required for ray.remote and ray.get
                            future_reward_result = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            # Compute synchronously, expect dict return
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)



                    # 4. Compute Log Probs (Actor & Ref Policy)
                    with _timer('old_log_prob', timing_raw):
                         old_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch)
                         # --- Entropy Loss (from new trainer) ---
                         if "entropys" in old_log_prob_output.batch:
                             entropys = old_log_prob_output.batch["entropys"]
                             response_masks = batch.batch["response_mask"] # Should exist now
                             loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                             entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                             metrics["actor/entropy_loss"] = entropy_loss.detach().cpu().item() # Use cpu()
                             old_log_prob_output.batch.pop("entropys")
                         # --- End Entropy Loss ---
                         batch = batch.union(old_log_prob_output)


                    if self.use_reference_policy:
                         with _timer('ref', timing_raw):
                             ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                             batch = batch.union(ref_log_prob)


                    # 5. Compute Values (Critic)
                    if self.use_critic:
                         with _timer('values', timing_raw):
                             values = self.critic_wg.compute_values(batch)
                             batch = batch.union(values)


                    # 6. Finalize Rewards & Compute Advantage
                    with _timer('adv', timing_raw):
                        # Get reward results (wait if async)
                        if self.config.reward_model.get("launch_reward_fn_async", False):
                             reward_result_async = ray.get(future_reward_result)
                             reward_tensor = reward_result_async["reward_tensor"]
                             reward_extra_infos_dict = reward_result_async.get("reward_extra_info", {})


                        # Store scores and extra info in batch
                        batch.batch['token_level_scores'] = reward_tensor
                        if reward_extra_infos_dict:
                            # Ensure values are numpy arrays for DataProto
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})


                        # Apply KL penalty (if enabled) using the function from new trainer
                        if self.config.algorithm.get("use_kl_in_reward", False): # Check config safely
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']


                        # Compute advantage using the function from new trainer
                        # *** ADD num_repeat ARGUMENT HERE ***
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n, # ADDED THIS ARG
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )


                    # 7. Update Critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        # Ensure metrics are detached and moved to CPU before reduction
                        critic_metrics_raw = critic_output.meta_info['metrics']
                        critic_metrics_processed = {k: v.detach().cpu().item() if torch.is_tensor(v) else v for k, v in critic_metrics_raw.items()}
                        critic_output_metrics = reduce_metrics(critic_metrics_processed)
                        metrics.update(critic_output_metrics)


                    # 8. Update Actor (respecting critic warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        # Ensure metrics are detached and moved to CPU before reduction
                        actor_metrics_raw = actor_output.meta_info['metrics']
                        actor_metrics_processed = {k: v.detach().cpu().item() if torch.is_tensor(v) else v for k, v in actor_metrics_raw.items()}
                        actor_output_metrics = reduce_metrics(actor_metrics_processed)
                        metrics.update(actor_output_metrics)
                    else:
                         metrics['training/skipped_actor_update_warmup'] = 1


                    # 9. Dump Rollout Generations (Optional, using inherited method)
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                         with _timer("dump_rollout_generations", timing_raw):
                             # Prepare data for dumping (decode, get scores)
                             inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                             outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                             # Use token_level_scores as the base for dumping
                             scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                             # Get reward extra info back from batch non_tensors
                             dump_reward_extra = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                                  for k,v in batch.non_tensor_batch.items()
                                                  if k in reward_extra_infos_dict} # Use dict from reward step


                             self._dump_generations(
                                 inputs=inputs,
                                 outputs=outputs,
                                 scores=scores,
                                 reward_extra_infos_dict=dump_reward_extra,
                                 dump_path=rollout_data_dir,
                             )


                    # 10. Validate (using overridden method)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                       (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate() # Use the overridden validation
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)


                    # 11. Save Checkpoint (using inherited method)
                    if self.config.trainer.save_freq > 0 and \
                       (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint() # Use inherited save


                # --- Post-Step ---
                # Add training step/epoch info
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })


                # Collect final batch metrics (use functions from new trainer)
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))


                # Log metrics
                logger.log(data=metrics, step=self.global_steps)


                # Check for termination
                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close() # Close tqdm bar
                    # Potentially call shutdown methods if needed
                    # self.shutdown()
                    return


                progress_bar.update(1) # Update tqdm bar
                self.global_steps += 1


            # End of epoch potentially reset dataloader if needed (depends on StatefulDataLoader)
            print(f"Epoch {epoch+1} completed.")

        progress_bar.close() # Close tqdm bar if loop finishes by epochs
        print("Training finished.")
        # Final validation and save if not done on last step
        if self.val_reward_fn is not None and (self.global_steps -1) % self.config.trainer.test_freq != 0:
             final_val_metrics = self._validate()
             pprint(f'Final validation metrics: {final_val_metrics}')
             logger.log(data=final_val_metrics, step=self.global_steps)
        if self.config.trainer.save_freq > 0 and (self.global_steps - 1) % self.config.trainer.save_freq != 0:
             self._save_checkpoint()

    # _save_checkpoint, _load_checkpoint, _maybe_log_val_generations, _dump_generations, _balance_batch
    # are inherited from the base RayPPOTrainer class.
