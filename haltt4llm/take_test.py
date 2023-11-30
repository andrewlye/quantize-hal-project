import argparse
import openai
import torch
import json
import time
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from autograd_4bit import load_llama_model_4bit_low_ram, load_wizardlm #Autograd4bitQuantLinear
from nf4 import load_nf4_model
# from peft import PeftModel
# from peft.tuners.lora import Linear4bitLt

def load_trivia_questions(file_path):
    with open(file_path, 'r') as file:
        trivia_data = json.load(file)
    return trivia_data

def generate_question_string(question_data):
    question = question_data['question']
    choices = [f"    {answer['choice']}. {answer['text']}\n" if answer != question_data['answers'][-1] else f"    {answer['choice']}. {answer['text']}" for answer in question_data['answers']]
    # return f"{question}\n{''.join(choices)}"

    return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {question} ASSISTANT:"""

    # return f"{question}\n It is ok to admit that you don't know. "

def grade_answers(question_data, llm_answer):
    correct_answer = None
    for answer in question_data['answers']:
        if answer['correct']:
            correct_answer = answer
            break

    if correct_answer is None:
        return "No correct answer found"

    normalized_llm_answer = llm_answer.lower().strip()
    normalized_correct_answer = correct_answer['text'].lower().strip()

    # lower case of the full text answer is in the llm's answer
    if normalized_correct_answer in normalized_llm_answer:
        return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " A." or  " B." or " C." or " D." or " E." for instance
    if f" {correct_answer['choice']}." in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " (A)" or  " (B)" or " (C)" or " (D)" or " (E)" for instance
    if f"({correct_answer['choice']})" in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    if "i don't know" in normalized_llm_answer or normalized_llm_answer == "d" or normalized_llm_answer == "d." or "i'm sorry" in normalized_llm_answer or "i'm not sure" in normalized_llm_answer:
        return f"{llm_answer} (uncertain)"

    return f"{llm_answer} (incorrect {correct_answer['choice']}.)"

def query_openai_gpt(prompt, engine):
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=50,
                temperature=0.1,
            )
            return response.choices[0].text.strip()
        except openai.error.RateLimitError as e:
            print("Rate limit exceeded. Pausing for one minute...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"Error: {e}")
            break

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPU(s) available.")
    device_index = 0
    if device_count > 1:
        device_index = input(f"Select device index (0-{device_count-1}): ")
        device_index = int(device_index)
    device = f"cuda:{device_index}"
    print(f"Using device: {device}")
else:
    device = "cpu"
    print("No GPU available, using CPU.")

def query_model(
        prompt,
        model,
        tokenizer,
        temperature=0.1,
        max_new_tokens=50,
        **kwargs,
    ):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=0.2,
            top_k=20,
            num_beams=1,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                do_sample=True,
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        response = output.split("ASSISTANT:")[1].strip()
        print("Model Output:", response.split("USER:")[0].strip()) #response.split("My Answer:")[1].strip()
        return  response #response.split("### Question:")[0].strip()

def main():
    parser = argparse.ArgumentParser(description='Run trivia quiz with GPT-3 or a local model.')
    parser.add_argument('--use-gpt3', action='store_true', help='Use GPT-3')
    parser.add_argument('--use-gpt3-5', action='store_true', help='Use GPT-3.5')
    parser.add_argument('--use-gpt4all', action='store_true', help='Use GPT4All')
    parser.add_argument('--use-llama', action='store_true', help='Use Llama')
    parser.add_argument('--openai-key', type=str, help='OpenAI API key')
    parser.add_argument('--trivia', type=str, help='File path to trivia questions')

    parser.add_argument('--model_dir', help='path of local model')
    parser.add_argument('--use_nf4',type=bool, default=False, help='test on nf4 quantization, will be run at time of initialization')
    parser.add_argument('--nf4_model',help='model to use for nf4 quantization')
    parser.add_argument('--nf4_cache',help='cache to use for nf4 quantization')
    args = parser.parse_args()

    use_gpt_3 = args.use_gpt3 or args.use_gpt3_5
    use_gpt4all = args.use_gpt4all

    if use_gpt_3 and use_gpt4all:
        print("Can't use both gpt and gpt4all at same time.")
        return

    if use_gpt_3 and not args.openai_key:
        print("Please provide an OpenAI API key with the --openai-key argument.")
        return

    if use_gpt_3:
        openai.api_key = args.openai_key

    if not use_gpt_3:
        if use_gpt4all:
            config_path = './models/llama-7b-hf/'
            model_path = './weights/llama-7b-4bit.pt'
            lora_path = './loras/gpt4all-lora/'
        else:
            config_path = './models/llama-7b-hf/'
            model_path = '/content/drive/MyDrive/llama-quant/llama-2-7b/gptq_model-4bit-128g.safetensors'   #'./weights/llama-7b-4bit.pt'
            lora_path = './loras/alpaca7B-lora/'

        
        if not args.use_nf4:
          model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype="auto", device_map="auto")
          tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
          model.to(device)
        else:
          model, tokenizer = load_nf4_model(args.nf4_model, args.nf4_cache)
        print(model.dtype)
        # model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path)
        # if not args.use_llama:
        #     model = PeftModel.from_pretrained(model, lora_path)
        #     print('Fitting 4bit scales and zeros to half')
        #     for n, m in model.named_modules():
        #         if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
        #             m.zeros = m.zeros.half()
        #             m.scales = m.scales.half()
        #             m.bias = m.bias.half()
    file_path = args.trivia
    trivia_data = load_trivia_questions(file_path)

    total_score = 0
    incorrect = []
    unknown = []

    if args.use_gpt3_5:
        model_name = "text-davinci-003"
    elif use_gpt_3:
        model_name = "text-davinci-002"
    elif args.use_llama:
        model_name = "llama-4bit"
    elif args.use_gpt4all:
        model_name = "gpt4all-4bit"
    else:
        model_name = "alpaca-lora-4bit"

    for i, question_data in enumerate(trivia_data):
        question_string = generate_question_string(question_data)
        prompt = question_string
        # prompt = generate_prompt(question_string)

        # print(f"Question {i+1}: {question_string}")
        if use_gpt_3:
            llm_answer = query_openai_gpt(prompt, model_name)
        else:
            llm_answer = query_model(prompt, model, tokenizer)

        answer_output = grade_answers(question_data, llm_answer)
        print(f"Answer: {answer_output}\n")

        if "(correct)" in answer_output:
            total_score += 2
        elif "(incorrect" in answer_output:
            incorrect.append((i+1, question_string, answer_output))
        else:
            total_score += 1
            unknown.append((i+1, question_string, answer_output))

    with open(f"test_results_{file_path}_{model_name}_4bit.txt", 'w') as f:
        f.write(f"Total score: {total_score} of {len(trivia_data) * 2}\n")
        i = len(incorrect)
        u = len(unknown)
        f.write(f"Correct: {len(trivia_data) - i - u}\n")
        if i:
            f.write(f"\nIncorrect: {i}\n")
            for question_num, question_string, answer_output in incorrect:
              try:
                f.write(f"Question {question_num}: {question_string.strip()}\n{answer_output.strip()}\n\n")
              except:
                continue
        if u:
            f.write(f"Unknown: {u}\n")
            for question_num, question_string, answer_output in unknown:
                f.write(f"Question {question_num}: {question_string.strip()}\n{answer_output.strip()}\n\n")

    print(f"Total score: {total_score} of {len(trivia_data) * 2}\n", end='')

def generate_prompt(instruction):
    # return f"""Below is an instruction that describes a task. Write a response that completes the request. Only answer the question. Keep token limit low.
    return f"""

### Question:
{instruction}

### Response:\n
"""
if __name__ == '__main__':
    main()

        # return f"""Here is an example of instruction and its response:
        # ### Instruction: What is the capital city of the USA?
        # A. San Francisco
        # B. Munich
        # C. Washington, D.C.
        # D. Beijing
        # ### Response:
        # C. Washington, D.C.

        # Below is an instruction that describes a task. Please write a response that completes the request.