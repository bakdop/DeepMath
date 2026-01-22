#!/bin/bash
#SBATCH --output=./log/%j.out # stdout file
#SBATCH --error=./log/%j.err # stderr file

#SBATCH --job-name=simpleRL-gkd
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --time=24:00:00
#SBATCH --mail-user=boyan9@ualberta.ca


module load cuda
source /home/biji/projects/aip-xiye17/biji/uvenvs/evmath/bin/activate

USER_ENV=`whoami`
set -x
export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1

# 定义必要的变量
WORKING_DIR=$(pwd)
# 动态获取当前节点的IP地址作为master节点
HEAD_IP=$(hostname -I | awk '{print $1}')
HEAD_PORT=8265
# 打印master节点信息，便于调试
echo "Ray master node IP: $HEAD_IP, Port: $HEAD_PORT"


# 环境变量设置
#/home/biji/projects/aip-xiye17/biji/models/Qwen3-1.7B-Base
export PROJECT_NAME=rl-gkd
export WANDB_API_KEY=aa7783a61740a28c4310d058e383e96dbc08cf97
export WANDB_OFFICIAL=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HDFS_DATA_PATH=/home/biji/projects/aip-xiye17/biji/simpleRL-reason/data
export HDFS_MODEL_PATH=/home/biji/projects/aip-xiye17/biji/models/Qwen3-1.7B-Base
export HDFS_REF_MODEL_PATH=/home/biji/projects/aip-xiye17/biji/models/Qwen3-8B
export HDFS_CHECKPOINT_PATH=./checkpoints
export HDFS_LOG_PATH=./log
export RUN_NAME=qwen3-1.7B-base-qwen3-8B-nonthinking-top20opd-noclamp # qwen3-1.7B-Base-RL
export ARNOLD_WORKER_NUM=1 # number of nodes you want to use

#/home/biji/projects/aip-xiye17/biji/LLaMA-Factory/saves/qwen3-1.7b-base/full/sft/checkpoint-160
# REF_MODEL_NAME=Qwen3-4B-Instruct-2507 #4B-Instruct-2507
# MODEL_NAME=Qwen3-1.7B-Base
# group_masked_opd
# response_masked_opd
# entropy_top20_opd 
# shaped_opd
# teacher_filtered_opd
# top10_posneg_opd
VAL_BEFORE_TRAIN=True
ADV_ESTIMATOR=top10_posneg_opd
KL_LOSS_COEF=0
TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=128
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096
LEARNING_RATE=1e-6
PPO_MINI_BATCH_SIZE=16
PPO_MICRO_BATCH_SIZE=1
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
CLIP_RATIO=0.2
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=1
ROLLOUT_N=5
KL_COEF=0.001
TOTAL_EPOCHS=1
DATASET_NAME=simplelr
ROLLOUT_GPU_MEMORY_UTIL=0.6
SAVE_FREQ=10
TEST_FREQ=5
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4
MICRO_ROLLOUT_BATCH_SIZE=1
REMOVE_PREVIOUS_CKPT=False
GRAD_CLIP=1.0


max_num_batched_tokens=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
# max_num_batched_tokens=2048
# 显示训练参数
echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Val Batch Size: $VAL_BATCH_SIZE"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "Learning Rate: $LEARNING_RATE"
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE"
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE"
echo "Micro Rollout Batch Size: $MICRO_ROLLOUT_BATCH_SIZE"
echo "KL Loss Coefficient: $KL_LOSS_COEF"
echo "KL Loss Type: $KL_LOSS_TYPE"
echo "Temperature: $TEMPERATURE"
echo "Rollout N: $ROLLOUT_N"
echo "KL Coefficient: $KL_COEF"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Dataset Name: $DATASET_NAME"
echo "Model Name: $MODEL_NAME"
echo "Remove Clip: $REMOVE_CLIP"
echo "Remove Previous Ckpt: $REMOVE_PREVIOUS_CKPT"

# 提交Ray作业
# ray job submit --address=${HEAD_IP}:${HEAD_PORT} \
#   --entrypoint-num-cpus=1 \
#   --runtime-env-json='{
#         "working_dir": "'${WORKING_DIR}'",
#         "env_vars": {
#           "http_proxy": "",
#           "https_proxy": ""
#         }
#     }' \
#   -- 
#  actor_rollout_ref.actor.optim.lr_warmup_steps=20 \

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=$ADV_ESTIMATOR \
  data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
  data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  actor_rollout_ref.model.path=$HDFS_MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
  actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
  actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
  actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.disable_log_stats=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.path=$HDFS_REF_MODEL_PATH \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=$KL_COEF \
  critic.ppo_micro_batch_size_per_gpu=1 \
  trainer.critic_warmup=0 \
  trainer.logger=[console,wandb] \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=$ARNOLD_WORKER_NUM \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
