import re
import torch
import argparse
from time import time
from tqdm import tqdm
from vllm import LLM, SamplingParams
from modelscope import AutoTokenizer, AutoModelForCausalLM
from vllm.lora.request import LoRARequest


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
        print("加载lora模型：", config.lora_path, max_model_len)
        llm = LLM(
            model=model_path, 
            gpu_memory_utilization=0.85,
            enable_lora=True,           # 必须开启！
            max_loras=1,                # 最多同时加载 4 个
            max_lora_rank=64,            # LoRA 秩上限
            max_model_len=max_seq_len,
            
        )
        lora_model = LoRARequest(
            lora_name="lora_model",
            lora_int_id=1,
            lora_path=config.lora_path
        )
    else:
        llm = LLM(
            model=model_path, 
            gpu_memory_utilization=0.85,
            max_model_len=max_seq_len,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def infer_vllm(prompt):
        sampling_params = SamplingParams(temperature=0, top_p=1.0, top_k=50, repetition_penalty=1.0, max_tokens=2048)
        SYSTEM_PROMPT = "you're a helpful assistant."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if config.lora_path:
            outputs = llm.generate(
                [text], sampling_params, 
                lora_request=lora_medical
            )
        else:
            outputs = llm.generate(
                [text], sampling_params
            )

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
        # print(generated_text)
        
        return generated_text

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

    true_num = 0
    for i in tqdm(range(len(dataset))):
        prompt = dataset[i]["question"]
        llm_answer = infer_vllm(prompt)
        llm_result = extract_number_from_boxed_string(llm_answer)
        #print(contains_boxed_structure(llm_answer))
        label = dataset[i]["answer"].replace(" ", "").replace(',', '')
        print((llm_result,label))
        if llm_result == label:
            print("True")
            true_num += 1
        print(f"---------{i}::true num:{true_num}/{len(dataset)}----------")


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
    config = parser.parse_args()
    test_vllm(config)   


if __name__ == '__main__':
    main()