from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser(description='GPTQ Quantization')
parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-chat-hf", help='model_name')
parser.add_argument('--cache_dir', default="/content/drive/MyDrive/quantize_hal_project/cache", help='cache dir')
parser.add_argument('--save_dir', default="/content/drive/MyDrive/quantize_hal_project/quant-models", help='save dir')
parser.add_argument('--bits', type=int, required=True, help='bits')
args = parser.parse_args()
print(args)

model_path = args.model_name
quant_path = args.save_dir
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": args.bits, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, cache_dir=args.cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
