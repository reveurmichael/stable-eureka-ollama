import ollama
response = ollama.chat(model='qwen:1.8b', messages=[
  {
    'role': 'user',
    'content': '你是谁?',
  },
])
print(response['message']['content'])