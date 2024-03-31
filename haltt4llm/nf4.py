import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
access_token = "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"

def load_nf4_model(model_id, cache_dir, access_token):
  tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=access_token)
  quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4")
  
  model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                token=access_token)
  
  return model, tokenizer
