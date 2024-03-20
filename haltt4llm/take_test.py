import argparse
import openai
import torch
import json
import time
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
from autograd_4bit import load_llama_model_4bit_low_ram, load_wizardlm #Autograd4bitQuantLinear
from nf4 import load_nf4_model
#from peft import PeftModel
#from peft.tuners.lora import Linear4bitLt

def load_trivia_questions(file_path):
    with open(file_path, 'r') as file:
        trivia_data = json.load(file)
    return trivia_data

def generate_question_string(question_data):
    question = question_data['question']
    choices = [f"{answer['choice']}. {answer['text']}\n" if answer != question_data['answers'][-1] else f"{answer['choice']}. {answer['text']}" for answer in question_data['answers']]
    prompt = f"Please carefully examine the weight matrix within the model, as it may contain errors. It is crucial to verify its accuracy and make any necessary adjustments to ensure optimal performance in the following scenario. An exchange between a user and a helpful assistant that provides correct answers to the multiple-choice trivia questions the user asks.\nUSER: {question}\n{''.join(choices)}\nASSISTANT:"
    return prompt

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
    if f"{correct_answer['choice']}." in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " (A)" or  " (B)" or " (C)" or " (D)" or " (E)" for instance
    if f"({correct_answer['choice']})" in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    if "i don't know" in normalized_llm_answer or "i'm sorry" in normalized_llm_answer or "i'm not sure" in normalized_llm_answer:
        return f"{llm_answer} (uncertain)"

    return f"{llm_answer} (incorrect, correct answer: {correct_answer['text']}.)"

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
        type,
        temperature=0.1,
        max_new_tokens=50,
        **kwargs,
    ):
        if (type == 'awq' or type == 'gptq' or type == 'nf4'):
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                generation_output = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens)
            output=tokenizer.decode(generation_output[0])
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
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
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )

            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
        print("Model Output: ", output)
        response = output.split("ASSISTANT: ")[1].split("USER:")[0]
        print("Detected Response: ", response)
        return  response #response.split("### Question:")[0].strip()

def main():
    parser = argparse.ArgumentParser(description='Run trivia quiz with GPT-3 or a local model.')
    # parser.add_argument('--use-gpt3', action='store_true', help='Use GPT-3')
    # parser.add_argument('--use-gpt3-5', action='store_true', help='Use GPT-3.5')
    # parser.add_argument('--use-gpt4all', action='store_true', help='Use GPT4All')
    # parser.add_argument('--use-llama', action='store_true', help='Use Llama')
    # parser.add_argument('--openai-key', type=str, help='OpenAI API key')
    
    parser.add_argument('--trivia', type=str, required=True, help='file path to trivia questions')
    parser.add_argument('--trivia-name', type=str, required=True, help='name of trivia test')
  
    parser.add_argument('--model_dir', help='path of local model')
    parser.add_argument('--model_name')

    parser.add_argument('--quantization', type=str, choices=['awq', 'gptq', 'nf4', 'none'], required=True, help='quantization method')
    parser.add_argument('--nf4_model',help='model to use for nf4 quantization')
    parser.add_argument('--nf4_cache',help='cache to use for nf4 quantization')
    parser.add_argument('--remote-path', help='path of remote model')
    parser.add_argument('--cache_dir', help='cache directory to use')
    parser.add_argument('--token', help='hf token', default='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')
    
    
    args = parser.parse_args()

    # use_gpt_3 = args.use_gpt3 or args.use_gpt3_5
    # use_gpt4all = args.use_gpt4all

    # if use_gpt_3 and use_gpt4all:
    #     print("Can't use both gpt and gpt4all at same time.")
    #     return

    # if use_gpt_3 and not args.openai_key:
    #     print("Please provide an OpenAI API key with the --openai-key argument.")
    #     return

    # if use_gpt_3:
    #     openai.api_key = args.openai_key

    # if not use_gpt_3 and not args.use_custom:
    #     if use_gpt4all:
    #         config_path = './models/llama-7b-hf/'
    #         model_path = './weights/llama-7b-4bit.pt'
    #         lora_path = './loras/gpt4all-lora/'
    #     else:
    #         config_path = './models/llama-7b-hf/'
    #         model_path = '/content/drive/MyDrive/llama-quant/llama-2-7b/gptq_model-4bit-128g.safetensors'   #'./weights/llama-7b-4bit.pt'
    #         lora_path = './loras/alpaca7B-lora/'
        
    #     model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path)
    #     if not args.use_llama:
    #         model = PeftModel.from_pretrained(model, lora_path)
    #         print('Fitting 4bit scales and zeros to half')
    #         for n, m in model.named_modules():
    #             if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
    #                 m.zeros = m.zeros.half()
    #                 m.scales = m.scales.half()
    #                 m.bias = m.bias.half()

    if args.quantization == 'awq':
        model = AutoAWQForCausalLM.from_quantized(args.model_dir, fuse_layers=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    elif args.quantization == 'gptq':
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto", torch_dtype=torch.float16)
    elif args.quantization == 'nf4':
        model, tokenizer = load_nf4_model(args.nf4_model, args.nf4_cache)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.remote_path,  device_map="auto", cache_dir=args.cache_dir, token=args.token)
        tokenizer = AutoTokenizer.from_pretrained(args.remote_path, token=args.token)
    
    file_path = args.trivia
    trivia_data = load_trivia_questions(file_path)

    total_score = 0
    incorrect = []
    unknown = []

    # if args.use_gpt3_5:
    #   model_name = "text-davinci-003"
    # elif use_gpt_3:
    #   model_name = "text-davinci-002"
    # elif args.use_llama:
    #   model_name = "llama-4bit"
    # elif args.use_gpt4all:
    #   model_name = "gpt4all-4bit"
    # elif not args.use_custom:
    #   model_name = "alpaca-lora-4bit"
    # else:
    
    model_name = args.model_name
      
    for i, question_data in enumerate(trivia_data):
        question_string = generate_question_string(question_data)
        prompt = question_string
        # prompt = generate_prompt(question_string)

        print(f"Question {i+1}: {question_string}")
        # if use_gpt_3:
        #     llm_answer = query_openai_gpt(prompt, model_name)
        # else:
        
        llm_answer = query_model(prompt, model, tokenizer, type=args.quantization)

        answer_output = grade_answers(question_data, llm_answer)
        print(f"Answer: {answer_output}\n")

        if "(correct)" in answer_output:
            total_score += 2
        elif "(incorrect" in answer_output:
            incorrect.append((i+1, question_string, answer_output))
        else:
            total_score += 1
            unknown.append((i+1, question_string, answer_output))

    with open(f"/results/{model_name}/{args.trivia_name}_{args.quantization}_test_results_4bit.txt", 'w') as f:
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
