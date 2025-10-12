from http.client import responses
from lib2to3.pgen2.tokenize import tokenize

from mlx_lm import  load, generate

model , tokenizer = load("mlx-community/Qwen3-4B-Instruct-2507-5bit-g32")

prompt = "hello, describe your self."

if tokenizer.chat_template is not None:
    messages = [{"role":"user","content":prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(response)