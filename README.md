# Post-Quantization Hallucination Tests for Language Models
This repository contains the necessary code to benchmark language model hallucination before and after quantization. Currently supported quantization methods are [GPTQ](https://arxiv.org/abs/2210.17323), Normalized Float 4, and [Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) (AWQ). To test for hallucination robustness, a modified version of [HALTT4LLM](https://github.com/manyoso/haltt4llm) is implemented.

## Installing Dependencies
```
pip install -r requirements.txt
```

## Quantizing a model
```
cd quantization
```
For GPTQ:
```
python run-gptq.py \
--model_name /path_to_local_or_HF_model \
--cache_dir /cache_directory \
--save_dir /where_to_save_quantized_model \
--bits  bits_to_quantize
```

For AWQ:
```
python run-awq.py \
--model_name /path_to_local_or_HF_model \
--cache_dir /cache_directory \
--save_dir /where_to_save_quantized_model \
--bits  bits_to_quantize
```

For NF4: \
Quantization is done at the same time the test is induced. Set use_nf4 to true, nf4_model to the model directory, and (optionally) nf4_cache to the cache directory.

## Administering the Hallucination Test
```
cd haltt4llm
python take_test.py \
--model_dir /path_to_model \
--model_name name_of_model \
--use_nf4 True/False \
--nf4_model /path_to_local_or_HF_model \
--nf4_cache
```
