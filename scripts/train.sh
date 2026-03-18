python ../src/main.py \
    --task=sft_train --bf16 --use_vllm \
    --model_name_or_path /root/autodl-tmp/model_hub/Qwen2.5-1.5B-Instruct \
    --checkpoint_dir ../outputs/Qwen2.5-1.5B-SFT_lora \
    --vllm_gpu_ratio 0.3 \
    --save_strategy steps \
    --epochs 2 \
    --learning_rate 5e-6 \
    --weight_decay 0.1

    #  sft_train   grpo_lora_train
    # /mnt/train1/lzl/model_hubs/Qwen2.5-3B-Instruct