python ../src/test_batch.py \
    --test_model_path /root/autodl-tmp/model_hub/Qwen2.5-3B-Instruct \
    --lora_path /root/autodl-tmp/lzl/qwen_grpo_rl/outputs/Qwen2.5-3B-SFT_lora/checkpoint-234 \
    --use_lora_path 1 \
    --batch_size 32
    
    # /mnt/train1/lzl/model_hubs/Qwen2.5-3B-Instruct
    # ../outputs/Qwen2.5-0.5B-SFT_lora/merged_qwen_model