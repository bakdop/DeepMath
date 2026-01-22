set -e
set -u

SCRIPT_DIR=/data/boyan/DeepMath/scripts
WORK_DIR=$SCRIPT_DIR/..
MODEL_DIR=$WORK_DIR/models
echo $SCRIPT_DIR
echo $WORK_DIR
echo $MODEL_DIR
DATA_DIR=/data/boyan/DeepMath/data
RUN_NAME=qwen3-1.7B-Base-qwen3-8B-nonthinking
mkdir -p $MODEL_DIR/$RUN_NAME

# ray job submit --address="http://127.0.0.1:8265" \
#     --runtime-env-json='{
#     "env_vars": {
#         "NCCL_NET_GDR_READ": "1",
#         "NCCL_IB_TIMEOUT": "24",
#         "NCCL_IB_GID_INDEX": "3",
#         "NCCL_IB_SL": "3",
#         "NCCL_CHECKS_DISABLE": "1",
#         "NCCL_P2P_DISABLE": "0",
#         "NCCL_IB_DISABLE": "0",
#         "NCCL_LL_THRESHOLD": "16384",
#         "NCCL_IB_CUDA_SUPPORT": "1",
#         "NCCL_SOCKET_IFNAME": "bond1",
#         "UCX_NET_DEVICES": "bond1",
#         "NCCL_IB_HCA": "mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6",
#         "NCCL_COLLNET_ENABLE": "0",
#         "SHARP_COLL_ENABLE_SAT": "0",
#         "NCCL_NET_GDR_LEVEL": "2",
#         "NCCL_IB_QPS_PER_CONNECTION": "4",
#         "NCCL_IB_TC": "160",
#         "NCCL_PXN_DISABLE": "1",
#         "GLOO_SOCKET_IFNAME": "bond1",
#         "VLLM_ATTENTION_BACKEND": "XFORMERS",
#         "PYTHONUNBUFFERED": "1",
#     },
#     "pip": ["word2number", "timeout_decorator"]
#     }' -- PYTHONUNBUFFERED=1 

export PYTHONPATH=$WORK_DIR:/home/user/uvenvs/dm/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_API_KEY=aa7783a61740a28c4310d058e383e96dbc08cf97
export WANDB_OFFICIAL=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HDFS_MODEL_PATH=/data/boyan/models/Qwen3-1.7B-Base
export HDFS_REF_MODEL_PATH=/data/boyan/models/Qwen3-8B
export HDFS_LOG_PATH=./log
export ARNOLD_WORKER_NUM=1 # number of nodes you want to use
export WANDB_RESUME=never


ADV_ESTIMATOR=on_policy_distillation
MAX_RESPONSE_LENGTH=8192
MODEL_PATH=/data/boyan/models/Qwen3-1.7B-Base
REF_MODEL_PATH=/data/boyan/models/Qwen3-8B
KL_LOSS_COEF=0.0
OVERLONG_ENABLE=False
OVERLONG_BUFFER=1024
PPO_MINI_BATCH_SIZE=16
python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/deepscaler_data/train.parquet \
    data.val_files="[$DATA_DIR/deepscaler_data/math.parquet,$DATA_DIR/deepscaler_data/aime.parquet]" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.rm_system_prompt=False \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.ref.path=$REF_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    custom_reward_function.path=$WORK_DIR/utils/reward_utils/reward_func.py \
    custom_reward_function.name=reward_func \
    custom_reward_function.overlong_buffer.enable=$OVERLONG_ENABLE \
    trainer.critic_warmup=0 \
    trainer.project_name=deepscaler \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$MODEL_DIR/$RUN_NAME \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.save_freq=100 \
    trainer.save_rollout=True \
    trainer.test_freq=5 \
    trainer.total_epochs=999999 \
    trainer.total_training_steps=1800 2>&1 | tee -a $MODEL_DIR/$RUN_NAME/train.log
    # trainer.resume_mode=disable \
    # trainer.resume_from_path=null \

# custom_reward_function.overlong_buffer.len=$OVERLONG_BUFFER \
# custom_reward_function.overlong_buffer.penalty_factor=1.0 \
# Llama-3.2-1B-Instruct
# Llama-3.1-Nemotron-Nano-8B-v1
# Qwen3-8B
# Qwen3-1.7B-Base


