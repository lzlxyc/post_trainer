python ../src/test_batch.py \
    --test_model_path /root/autodl-tmp/model_hub/Qwen2.5-1.5B-Instruct \
    --lora_path /root/autodl-tmp/lzl/qwen_grpo_rl/outputs/Qwen2.5-1.5B-SFT_lora/checkpoint-468 \
    --use_lora_path 1 \
    --batch_size 64
    
    # /mnt/train1/lzl/model_hubs/Qwen2.5-3B-Instruct
    # ../outputs/Qwen2.5-0.5B-SFT_lora/merged_qwen_model