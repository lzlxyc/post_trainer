import os
import torch
import argparse
from peft import PeftModel
from modelscope import AutoTokenizer, AutoModelForCausalLM


def merge_lora(
    base_model_path,
    lora_model_path,
    output_dir,
    bf16=True,
    safe_serialization=True,
    device_map="auto"
):
    """
    合并LoRA权重到基础模型
    
    Args:
        base_model_path: 基础模型的路径/名称（如Qwen/Qwen2.5-7B）
        lora_model_path: 训练好的LoRA权重路径（通常是checkpoint目录）
        output_dir: 合并后模型的保存路径
        bf16: 是否使用bf16精度保存（与训练保持一致）
        safe_serialization: 是否启用安全序列化（避免大文件问题）
        device_map: 设备映射策略（auto自动分配，cpu避免GPU显存不足）
    """
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 输出目录：{output_dir}")

    # 2. 加载基础模型
    print(f"[INFO] 加载基础模型：{base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map=device_map,
        trust_remote_code=True,  # 加载自定义模型时需要
    )

    # 3. 加载LoRA适配器
    print(f"[INFO] 加载LoRA权重：{lora_model_path}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        dtype=torch.bfloat16 if bf16 else torch.float32,
    )

    # 4. 合并LoRA权重并卸载适配器
    print("[INFO] 开始合并LoRA权重...")
    merged_model = lora_model.merge_and_unload()

    # 5. 加载并保存tokenizer
    print("[INFO] 保存Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side="right",
        truncation_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)

    # 6. 保存合并后的模型
    print(f"[INFO] 保存合并后的模型到：{output_dir}")
    merged_model.save_pretrained(
        output_dir,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        safe_serialization=safe_serialization,
    )

    print("[SUCCESS] 模型合并完成！")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="合并LoRA权重到基础模型")
    
    # 添加命令行参数
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/mnt/train1/lzl/model_hubs/Qwen2.5-0.5B-Instruct",
        help="基础模型路径（本地路径或ModelScope/HuggingFace名称）"
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="outputs/Qwen2.5-0.5B-GRPO_lora_v3/checkpoint-500",
        help="训练好的LoRA权重路径（如checkpoint-500）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/Qwen2.5-0.5B-GRPO_lora_v3/merged_qwen_model",
        help="合并后模型保存路径"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="是否使用bf16精度保存（默认开启）"
    )
    parser.add_argument(
        "--no-bf16",
        action="store_false",
        dest="bf16",
        help="关闭bf16精度保存"
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        default=False,
        help="是否仅使用CPU合并（默认关闭）"
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="是否启用安全序列化（默认开启）"
    )
    parser.add_argument(
        "--no-safe-serialization",
        action="store_false",
        dest="safe_serialization",
        help="关闭安全序列化"
    )
    
    return parser.parse_args()


def main():
    # 加载配置
    config = parse_args()
    # 设置设备映射
    device_map = "cpu" if config.cpu_only else "auto"
    # 执行合并
    merge_lora(
        base_model_path=config.base_model_path,
        lora_model_path=config.lora_model_path,
        output_dir=config.output_dir,
        bf16=config.bf16,
        safe_serialization=config.safe_serialization,
        device_map=device_map
    )


if __name__ == "__main__":
    main()