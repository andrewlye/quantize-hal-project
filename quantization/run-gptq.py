import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, LlamaTokenizer
import argparse

parser = argparse.ArgumentParser(description='GPTQ Quantization')
parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-chat-hf", help='model_name')
parser.add_argument('--cache_dir', help='cache dir')
parser.add_argument('--save_dir', default="/content/drive/MyDrive/quantize_hal_project/save", help='save dir')
parser.add_argument('--bits', type=int, required=True, help='bits')
args = parser.parse_args()
print(args)

access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(args.model_name, token = access_token)
quantization_config = GPTQConfig(bits=args.bits, group_size=128, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", 
device_map="auto", quantization_config=quantization_config, token = access_token, cache_dir=args.cache_dir)

model.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
