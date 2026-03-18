from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from modelscope import AutoTokenizer, AutoModelForCausalLM


from utils import SYSTEM_PROMPT


def infer(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # prompt = "Xiao Ming bought 4 apples, ate 1, and gave 1 to his sister. How many apples were left?"
    while True:
        print("请输入你的问题：")
        prompt = input()

        if prompt in ("exit", "bye"):
            print("Assistant: 再见👋")
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_completion_length,
            temperature=0
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Assistant:\n{response}")

        
def infer_vllm(args):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(
        temperature=0, top_p=1.0, top_k=50, 
        repetition_penalty=1.0, max_tokens=None
    )

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=args.checkpoint_dir, gpu_memory_utilization=0.4)
    #llm = LLM(model="outputs/Qwen-0.5B-SFT-FirstHalf/checkpoint-233", gpu_memory_utilization=0.4)

    # Prepare your prompts
    prompt = "Gretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does Gretchen have?"

    SYSTEM_PROMPT = """
    Respond in the following format:
    <think>
    ...
    </think>
    <answer>
    ...
    </answer>
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # generate outputs
    outputs = llm.generate([text], sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(generated_text)