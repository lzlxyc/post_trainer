import os
import torch
from trl import GRPOConfig, GRPOTrainer
from modelscope import AutoTokenizer, AutoModelForCausalLM
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


from utils import get_gsm8k_dataset
from rewards import reward_funcs_map


def train(args):
    
    # from transformers import BitsAndBytesConfig
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,                  # 4位量化加载
    #     bnb_4bit_compute_dtype=torch.float16, # 计算 dtype
    #     bnb_4bit_use_double_quant=True,     # 双量化
    #     bnb_4bit_quant_type="nf4"           # 量化类型（nf4更适合LLM）
    # )

    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16, #if args.bf16 else None,
        device_map=None,
        # quantization_config=bnb_config,  # 应用量化配置
        cache_dir=args.cache_dir
    )

    lora_config = LoraConfig(
        r=args.lora_r,  # LoRA秩
        lora_alpha=args.lora_alpha,  # LoRA alpha参数
        target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else [
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        ],
        lora_dropout=args.lora_dropout,  # Dropout率
        bias="none",  # 不训练bias
        task_type="CAUSAL_LM",  # 任务类型
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数数量
    
    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",  # 右padding，避免推理时的显存碎片
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
        output_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        max_grad_norm=args.max_grad_norm,
        log_on_each_node=False,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_ratio,
        report_to="none",
        save_total_limit=5,  
        reward_weights=[0.2, 0.8],
        multi_objective_aggregation="normalize_then_sum",

        gradient_checkpointing=False,  # 全局启用梯度检查点
        remove_unused_columns=False,  # 不删除未使用的列，减少数据处理显存占用
        dataloader_pin_memory=False,  # 关闭pin_memory，节省GPU显存（会轻微降速）
    )

    reward_funcs = [reward_funcs_map[func.strip()] for func in args.reward_funcs.split(',')]

    experiment_name = args.checkpoint_dir.split('/')[-1]
    print(f"Experiment name: {experiment_name}")
    swanlab_callback = SwanLabCallback(
        project="lzl", 
        experiment_name=experiment_name
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=get_gsm8k_dataset(
            cache_dir=args.cache_dir,
            first_half=args.split_half == "first_half",
            second_half=args.split_half == "second_half"
        ),
        callbacks=[swanlab_callback],
    )
    
    # 开始训练
    trainer.train()
    
    # 训练结束后保存LoRA权重
    if hasattr(args, 'save_lora_path') and args.save_lora_path:
        model.save_pretrained(args.save_lora_path)
        tokenizer.save_pretrained(args.save_lora_path)
        print(f"LoRA weights saved to {args.save_lora_path}")