# verl/trainer/main_ppo_async.py
# Entry point for the asynchronous PPO trainer

import os
import hydra
import ray
from omegaconf import OmegaConf
from pprint import pprint

# Import the modified async trainer
from verl.trainer.ppo.ray_trainer_async import RayPPOAsyncTrainer

# Import necessary components from the NEW standard library structure
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, AdvantageEstimator # Added AdvantageEstimator
from verl.trainer.ppo.reward import load_reward_manager # Use the new reward loader
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local # Use the standard copy function
from verl.single_controller.ray import RayWorkerGroup # Default worker group class

# Import worker classes (adjust paths if necessary in your verl structure)
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker, # Import the async worker
    CriticWorker,
    RewardModelWorker
)
# Add Megatron imports if needed


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    """Main function to launch the PPO training pipeline."""
    run_ppo_pipeline(config)

def run_ppo_pipeline(config):
    """Sets up Ray and launches the main training task."""
    # Environment setup (copied from new standard main)
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN', # Or desired level
                    'VLLM_LOGGING_LEVEL': 'WARN' # Or desired level
                }
            },
             # Add num_cpus from config if specified (like new standard main)
             num_cpus=config.ray_init.get("num_cpus", None)
        )

    # Use TaskRunner pattern like the new standard main for better isolation
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1) # Ensure runner doesn't conflict with worker resources
class TaskRunner:
    """Encapsulates the main training setup and execution within a Ray remote actor."""

    def run(self, config):
        """Sets up workers and trainer, then starts training."""
        print("--- Initial Configuration ---")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        print("-----------------------------")

        # --- Setup Model Path, Tokenizer, Processor ---
        print("Copying model data locally...")
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        print(f"Model data available at: {local_path}")

        print("Loading tokenizer and processor...")
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True) # Optional: for multi-modal
        print("Tokenizer and processor loaded.")

        # --- Define Worker Classes and Resource Mapping ---
        print("Defining worker classes and resource mapping...")
        # Determine ActorRollout class based on config
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            print("Using FSDP strategy.")
            # Select worker based on rollout mode config
            actor_rollout_cls = AsyncActorRolloutRefWorker if OmegaConf.select(config, "actor_rollout_ref.rollout.mode") == "async" else ActorRolloutRefWorker
            critic_cls = CriticWorker
            ref_policy_cls = ActorRolloutRefWorker # Typically standard worker for ref policy
            rm_cls = RewardModelWorker
            ray_worker_group_cls = RayWorkerGroup
        # Add elif for 'megatron' if needed
        else:
            raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        # Define resource pool (typically single global pool for hybrid)
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        # Determine which optional components are needed based on config
        # Use Critic? (Based on advantage estimator)
        # Ensure AdvantageEstimator is imported or defined
        use_critic = config.algorithm.adv_estimator == AdvantageEstimator.GAE
        if use_critic:
            print("Critic enabled.")
            role_worker_mapping[Role.Critic] = ray.remote(critic_cls)
            mapping[Role.Critic] = global_pool_id

        # Use Reference Policy? (Based on KL usage)
        use_reference_policy = OmegaConf.select(config, "algorithm.use_kl_in_reward", default=False) or \
                               OmegaConf.select(config, "actor_rollout_ref.actor.use_kl_loss", default=False)
        if use_reference_policy:
            print("Reference Policy enabled.")
            role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            mapping[Role.RefPolicy] = global_pool_id

        # Use Reward Model?
        use_rm = config.reward_model.enable
        if use_rm:
            print("Reward Model enabled.")
            role_worker_mapping[Role.RewardModel] = ray.remote(rm_cls)
            mapping[Role.RewardModel] = global_pool_id

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        print("Resource pool manager created.")

        # --- Load Reward Function ---
        print("Loading reward functions...")
        # Use the standard loader function
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1) # Validation typically uses default kwargs
        print("Reward functions loaded.")


        # --- Instantiate Trainer ---
        print("Instantiating RayPPOAsyncTrainer...")
        trainer = RayPPOAsyncTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )
        print("Trainer instantiated.")

        # --- Initialize Workers and Start Training ---
        print("Initializing workers...")
        trainer.init_workers() # This now initializes the async manager if needed
        print("Workers initialized.")

        print("Starting training...")
        trainer.fit()
        print("Training finished.")

        # Optional: Add shutdown logic if needed
        # print("Shutting down...")
        # trainer.shutdown() # If trainer has a shutdown method
        # ray.shutdown()


if __name__ == '__main__':
    main()

