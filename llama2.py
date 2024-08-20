import requests

def ask_llama2(question):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama2:7B",
        "prompt": question,
        "max_tokens": 256,
        "temperature": 0.7
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["text"]
    else:
        return f"Error: {response.status_code}"

# 从文件中读取问题
with open("questions.txt", "r") as file:
    for line in file:
        answer = ask_llama2(line.strip())
        print(f"Answer: {answer}")