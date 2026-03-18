import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from utils import get_gsm8k_dataset


def train(args):
    training_args = SFTConfig(
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
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=5,  
        max_grad_norm=args.max_grad_norm,
        log_on_each_node=False,
        report_to="none", # report_to="swanlab"
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # load_in_8bit=True,
        dtype=torch.bfloat16 if args.bf16 else None,
        device_map='auto',
        cache_dir=args.cache_dir
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数数量
    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    experiment_name = args.checkpoint_dir.split('/')[-1]
    print(f"Experiment name: {experiment_name}")
    swanlab_callback = SwanLabCallback(
        project="lzl", 
        experiment_name=experiment_name
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=get_gsm8k_dataset(sft=True, cache_dir=args.cache_dir,
                                        first_half=args.split_half=="first_half",
                                        second_half=args.split_half=="second_half"),
        callbacks=[swanlab_callback],
    )
    trainer.train()
