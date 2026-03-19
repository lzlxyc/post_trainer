python ../src/main.py \
    --task=sft_train --bf16 --use_vllm \
    --model_name_or_path /root/autodl-tmp/model_hub/Qwen3-4B-Instruct-2507 \
    --checkpoint_dir ../outputs/Qwen3-4B-SFT_lora \
    --vllm_gpu_ratio 0.35 \
    --save_strategy steps \
    --epochs 1 \
    --learning_rate 5e-6 \
    --weight_decay 0.1

    #  sft_train   grpo_lora_train
    # /mnt/train1/lzl/model_hubs/Qwen2.5-3B-Instruct