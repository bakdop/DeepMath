VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 
VLLM_ATTENTION_BACKEND=XFORMERS
VLLM_USE_V1=1
VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Eval base model"
# python3 uni_eval.py \
#     --base_model /data/boyan/DeepMath/models/qwen3-1.7B-Base-qwen3-8B-nonthinking/global_step_250_deepmath/actor/huggingface \
#     --chat_template_name default \
#     --system_prompt_name simplerl \
#     --output_dir /data/boyan/DeepMath/evaluation/qwen3-1.7B-Base \
#     --bf16 True \
#     --tensor_parallel_size 4 \
#     --data_id zwhe99/aime90 \
#     --split 2024 \
#     --max_model_len 32768 \
#     --temperature 0.7 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --n 16

# echo "Eval rl model"
# python3 uni_eval.py \
#     --base_model /data/boyan/DeepMath/models/qwen3-1.7B-Base-qwen3-8B-nonthinking-RL/global_step_250_deepmath_rl/actor/huggingface \
#     --chat_template_name default \
#     --system_prompt_name simplerl \
#     --output_dir /data/boyan/DeepMath/evaluation/qwen3-1.7B-qwen3-8B-global_step_250_deepmath_rl \
#     --bf16 True \
#     --tensor_parallel_size 4 \
#     --data_id zwhe99/aime90 \
#     --split 2024 \
#     --max_model_len 32768 \
#     --temperature 0.7 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --n 16


# echo "Eval gkd model"
python3 uni_eval.py \
    --base_model /data/boyan/DeepMath/models/qwen3-1.7B-Base-qwen3-8B-nonthinking/global_step_250_deepmath/actor/huggingface \
    --chat_template_name default \
    --system_prompt_name simplerl \
    --output_dir /data/boyan/DeepMath/evaluation/qwen3-1.7B-qwen3-8B-global_step_250_deepmath-gkd \
    --bf16 True \
    --tensor_parallel_size 4 \
    --data_id zwhe99/aime90 \
    --split 2024 \
    --max_model_len 32768 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --n 16