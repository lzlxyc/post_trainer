import re
import torch
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from modelscope import AutoTokenizer, AutoModelForCausalLM


from utils import SYSTEM_PROMPT
from utils import get_gsm8k_dataset


def test_vllm(config):
    model_path = config.test_model_path
    dataset = get_gsm8k_dataset(split='test').take(100)
    # print(dataset)
    # input(f"数据量：{len(dataset)}")

    if not config.use_lora_path:
        config.lora_path = None

    max_seq_len = 4096*4
    if config.lora_path:
        print("加载lora模型：", config.lora_path)
        llm = LLM(
            model=model_path, 
            gpu_memory_utilization=0.9,
            enable_lora=True,           # 必须开启！
            max_loras=1,                # 最多同时加载 4 个
            max_lora_rank=64,            # LoRA 秩上限    
        )
        lora_model = LoRARequest(
            lora_name="lora_model",
            lora_int_id=1,
            max_model_len=max_seq_len
        )
    else:
        llm = LLM(
            model=model_path, 
            gpu_memory_utilization=0.9,
            max_model_len=max_seq_len,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def infer_vllm(prompts):
        sampling_params = SamplingParams(temperature=0, top_p=1.0, top_k=50, repetition_penalty=1.0, max_tokens=2048)
        SYSTEM_PROMPT = "you're a helpful assistant."
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ] for prompt in prompts
        ]
        texts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # generate outputs
        if config.lora_path:
            outputs = llm.generate(
                texts, sampling_params, 
                lora_request=lora_model
            )
        else:
            outputs = llm.generate(
                texts, sampling_params
            )
        # Print the outputs.
        generated_texts = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        # print(generated_text)
        
        return generated_texts

    def extract_number_from_boxed_string(s):
        # 修改正则表达式以匹配可能存在的货币符号、逗号和数字
        s = s.replace('\\!', '')
        number = re.search(r'boxed[^\d]*(\d[\d,]*)', s)
        # 提取数字并移除逗号
        extractednumber = number.group(1).replace(',', '') if number else None
        return extractednumber

    def contains_boxed_structure(s):
        # 使用正则表达式来匹配boxed{...}结构
        pattern = r'boxed\{[^}]*\}'
        if re.search(pattern, s):
            return 1
        else:
            return 0

    start = time()


    # 设置批处理大小
    batch_size = config.batch_size  # 可以根据显存大小调整
    true_num = 0

    # 批量处理数据
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch_prompts = [dataset[idx]["question"] for idx in batch_indices]
        
        # 批量推理
        llm_answers = infer_vllm(batch_prompts)  # 返回list
        
        # 处理批量结果
        for j, idx in enumerate(batch_indices):
            llm_answer = llm_answers[j]
            llm_result = extract_number_from_boxed_string(llm_answer)
            label = dataset[idx]["answer"].replace(" ", "").replace(',', '')
            
            print((llm_result, label))
            if llm_result == label:
                print("True")
                true_num += 1
            
            print(f"---------{idx}::true num:{true_num}/{len(dataset)}----------")

    print(f"=========acc:{round(true_num/len(dataset), 4)}  || **** time:{round(time()-start, 4)}=========")


def main():
    parser = argparse.ArgumentParser(description="模型测试")
    parser.add_argument(
        "--test_model_path",
        type=str,
        default="/mnt/train1/lzl/model_hubs/Qwen2.5-0.5B-Instruct",
        help="模型测试"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="模型测试"
    )
    parser.add_argument(
        "--use_lora_path",
        type=int,
        default=0,
        help="模型测试"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="模型测试"
    )
    config = parser.parse_args()
    test_vllm(config)   


if __name__ == '__main__':
    main()