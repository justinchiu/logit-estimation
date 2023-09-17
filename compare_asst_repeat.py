from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from torch.distributions import Categorical, kl_divergence

torch.set_grad_enabled(False)

llama = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama)
llama_model = AutoModelForCausalLM.from_pretrained(llama)


inputs = llama_tokenizer(
    ["[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n[INST] Hi [/INST] Hi. How can I"],
    return_tensors="pt",
)
outputs = llama_model.generate(
    **inputs,
    max_new_tokens=2,
    return_dict_in_generate=True,
    output_scores=True,
    do_sample=True,
    num_return_sequences=1,
)
true_dist = Categorical(logits=outputs.scores[0][0])

inputs = llama_tokenizer(
    ["[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n[INST] Hi [/INST] Hi. How can I [/INST]"],
    return_tensors="pt",
)
outputs2 = llama_model.generate(
    **inputs,
    max_new_tokens=2,
    return_dict_in_generate=True,
    output_scores=True,
    do_sample=True,
    num_return_sequences=1,
)
dist = Categorical(logits=outputs2.scores[0][0])


sample_kl = kl_divergence(true_dist, dist)
print(sample_kl)

print(llama_tokenizer.batch_decode(outputs.sequences))
print(llama_tokenizer.batch_decode(outputs2.sequences))

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi."},
  ],
  temperature=0,
  max_tokens=1024
)
print(response)

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hi."},
  ],
  temperature=0,
  max_tokens=1024
)
print(response)

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hi. How can I help"},
  ],
  temperature=0,
  max_tokens=1024
)
print(response)

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hi. How can I help"},
  ],
  temperature=0,
  max_tokens=1024
)
print(response)
import pdb; pdb.set_trace()
