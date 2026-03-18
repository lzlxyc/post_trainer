python ../src/merge.py \
    --base_model_path /mnt/train1/lzl/model_hubs/Qwen2.5-3B-Instruct \
    --lora_model_path ../outputs/Qwen2.5-3B-GRPO_lora/checkpoint-400 \
    --output_dir ../outputs/Qwen2.5-3B-GRPO_lora/merged_qwen_model \
    --safe_serialization