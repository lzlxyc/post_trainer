python ../src/test_batch.py \
    --test_model_path /root/autodl-tmp/model_hub/Qwen3-4B-Instruct-2507 \
    --lora_path /root/autodl-tmp/lzl/post_trainer/outputs/Qwen3-4B-SFT_lora/checkpoint-200 \
    --use_lora_path 0 \
    --batch_size 16
    
    # /mnt/train1/lzl/model_hubs/Qwen2.5-3B-Instruct
    # ../outputs/Qwen2.5-0.5B-SFT_lora/merged_qwen_model