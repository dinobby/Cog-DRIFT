set -x
export WANDB_PROJECT=Cog-DRIFT
export WANDB_EXP=Qwen4B-Adaptive-Curriculum

# HF model/dataset cache — override with your own path if needed
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)

TRAIN_DATA=${REPO_ROOT}/processed_data/BMH_Adaptive/train.parquet
VAL_DATA=${REPO_ROOT}/processed_data/BMH_Adaptive/val.parquet
VARIANTS_LOOKUP=${REPO_ROOT}/processed_data/BMH_Adaptive/variants_lookup.json
MODEL_PATH=Qwen/Qwen3-4B-Instruct

# Adaptive curriculum dataset & sampler classes
ADAPTIVE_DATASET_PATH=${REPO_ROOT}/verl/utils/dataset/adaptive_rl_dataset.py
ADAPTIVE_SAMPLER_PATH=${REPO_ROOT}/verl/experimental/dataset/adaptive_sampler.py

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=8 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.truncation=right \
    +data.adaptive=True \
    +data.variants_lookup=${VARIANTS_LOOKUP} \
    +data.adaptive_threshold=0.5 \
    +data.adaptive_reward_window=1 \
    data.custom_cls.path=${ADAPTIVE_DATASET_PATH} \
    data.custom_cls.name=AdaptiveRLHFDataset \
    data.sampler.class_path=${ADAPTIVE_SAMPLER_PATH} \
    data.sampler.class_name=AdaptiveCurriculumSampler \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=30000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_EXP \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.balance_batch=True \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100 \
    +trainer.format_reward=True \
    +trainer.format_reward_coef=0.2 "${@:1}"
