#!/bin/bash
#SBATCH --output=./log/%j.out # stdout file
#SBATCH --error=./log/%j.err # stderr file
#SBATCH --job-name=deepmath-trrd-llama
#SBATCH --nodes=1
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:h20:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=48:00:00
#SBATCH --mail-user=boyan9@ualberta.ca

source /scratch/bl4363/uvenvs/dm/bin/activate

set -e
set -u


WORK_DIR=/scratch/bl4363/DeepMath
MODEL_DIR=$WORK_DIR/models
DATA_DIR=/scratch/bl4363/DeepMath/data
RUN_NAME=llama3.2-1B-Instruct-nemotron-nano-8B-trrd
mkdir -p $MODEL_DIR/$RUN_NAME

export WANDB_API_KEY=aa7783a61740a28c4310d058e383e96dbc08cf97
export WANDB_OFFICIAL=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export ARNOLD_WORKER_NUM=1

ADV_ESTIMATOR=grpo
TRRD_ENABLED=true
TRRD_ALPHA=0.5
MAX_RESPONSE_LENGTH=8192
MODEL_PATH=/scratch/bl4363/models/Llama-3.2-1B-Instruct
REF_MODEL_PATH=/scratch/bl4363/models/Llama-3.1-Nemotron-Nano-8B-v1
KL_LOSS_COEF=0.0
OVERLONG_ENABLE=False
OVERLONG_BUFFER=1024
PPO_MINI_BATCH_SIZE=64
python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.rm_system_prompt=False \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    +data.llama_nemotron_compat=True \
    +data.teacher_chat_template_path=$REF_MODEL_PATH \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.ref.path=$REF_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.trrd_enabled=$TRRD_ENABLED \
    actor_rollout_ref.actor.trrd_alpha=$TRRD_ALPHA \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    custom_reward_function.path=$WORK_DIR/utils/reward_utils/reward_func.py \
    custom_reward_function.name=reward_func \
    custom_reward_function.overlong_buffer.enable=$OVERLONG_ENABLE \
    trainer.critic_warmup=0 \
    trainer.project_name=deepmath-llama \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$MODEL_DIR/deepmath-llama/$RUN_NAME \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.save_freq=10 \
    trainer.save_rollout=True \
    trainer.test_freq=2 \
    trainer.total_epochs=999999 \
    trainer.total_training_steps=1800 2>&1 | tee -a $MODEL_DIR/$RUN_NAME/train.log
