import os
path = "./results/Llama-2-13b-chat-hf"

if not os.path.exists(path):
  os.makedirs(path)

with open(f"./results/Llama-2-13b-chat-hf/test_1_test_results_4bit.txt", 'w') as f:
    f.write(f"Total score:")
