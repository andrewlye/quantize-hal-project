import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import argparse
import json
os.environ['HOME'] = '/scratch/zx22/andrew'

parser = argparse.ArgumentParser(description='GPTQ Quantization')
parser.add_argument('--model-name', default="meta-llama/Llama-2-13b-chat-hf", help='model_name')
parser.add_argument('--cache-dir', default="/scratch/zx22/andrew/quantize-hal-project/cache", help='cache dir')
parser.add_argument('--save-dir', default="/scratch/zx22/andrew/quantize-hal-project/quant-models", help='save dir')
parser.add_argument('--bits', type=int, required=True, help='bits')
parser.add_argument('--token', default='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')
args = parser.parse_args()
print(args)

save_folder = f'{args.save_dir}/gptq'

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')

quantizer = GPTQQuantizer(bits=args.bits, dataset="wikitext2", model_seqlen = 2048)
quantized_model = quantizer.quantize_model(model, tokenizer)

quantized_model.save_pretrained(save_folder, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(model_name).save_pretrained(save_folder)

with open(os.path.join(save_folder, "quantize_config.json"), "w", encoding="utf-8") as f:
  quantizer.disable_exllama = False
  json.dump(quantizer.to_dict(), f, indent=2)