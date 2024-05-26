# Stable Eureka
Stable Eureka is an iterative llm-based reward designer for reinforcement learning. It integrates
stable-baselines3, open-source LLMs and gym-based environments. 


## Installation

```bash
git clone https://github.com/rsanchezmo/stable-eureka.git
cd stable-eureka
pip install .
```

You must install ollama before running the code:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Available LLMs
You have to pull the LLMs you want to use from the ollama repository. For example, to pull the llama3 LLM:
```bash
ollama pull llama3
```

- llama3
- codellama
- mistral
- phi3
- gemma