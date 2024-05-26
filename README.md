# Stable Eureka
Stable Eureka is an iterative llm-based reward designer for reinforcement learning. It integrates
stable-baselines3, open-source LLMs and gym-based environments. This repo is based on [NVIDIA Eureka](https://github.com/eureka-research/Eureka/tree/main).


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

## Configuration
You must fill a configuration file with the following structure:
```yaml
eureka:
    model: 'llama3'
    temperature: 0.5
    iterations: 1
    samples: 3

environment:
    name: 'bipedal_walker'

experiment:
    parent: 'experiments'
    name: 'bipedal_walker'
    use_datetime: true
```