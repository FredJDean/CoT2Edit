export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# others 4 7 16
# num_processes = 7 is that one GPU is used to vllm inference, it can also 8 and frozen vllm
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=7 src/open_r1/grpo.py \
  --output_dir ./log/grpo_Llama3 \
  --model_name_or_path /your_model_path/Meta-Llama-3-8B-Instruct \
  --dataset_name /your_data_path/train_grpo.jsonl \
  --max_prompt_length 512 \
  --max_completion_length 1024 \
  --per_device_train_batch_size 4 \
  --num_generations 7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --logging_strategy steps \
  --learning_rate 3.0e-06 \
  --gradient_accumulation_steps 16 \
  --logging_steps 10 \
  --eval_strategy no \
  --bf16 True \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.9