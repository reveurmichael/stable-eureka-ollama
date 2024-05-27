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

## Environment
You must provide the env code in a `env.py` file for now. You should include take the step func into a `step.py` file, and must
create a `task_description.txt` file with the task description:

```
envs/
    bipedal_walker/
        env.py
        step.py
        task_description.txt
```

The code will copy the code into the experiments folder and append the reward function to it. The reward function should 
satisfy the signature:
```python
reward, intermediate_reward = self.reward_function(param1, param2, param3)
```
By doing so, the code will be automatically executed by the experiment runner once the reward function is appended.


## TODO:
- run parallel processes and training with stable baselines
- evaluate the trained agent and compute fitness scores
- create the reward reflection code
- add some plots of fitness scores during iterations