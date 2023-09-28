"""
Gpt-4 sometimes ignores the last assistant turn, while gpt-3.5 completes it.
"""

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
