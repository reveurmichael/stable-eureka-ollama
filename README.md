# Stable Eureka
Stable Eureka is an iterative llm-based reward designer for reinforcement learning. It integrates
[stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/), open-source LLMs (running locally with [ollama](https://www.ollama.com/)) and [gymnasium](https://gymnasium.farama.org/)-based environments. This repo is based on [NVIDIA Eureka](https://github.com/eureka-research/Eureka/tree/main).

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
### Ollama
You have to pull the LLMs you want to use from the ollama repository. For example, to pull the llama3 LLM:
```bash
ollama pull llama3
```
- `llama3`: very fast model, not very accurate 7B
- `codestral`: too large for my gpu (`rtx4070 8gb` too slow)
- `codellama`: a bit slower than llama3
- `mistral`
- `phi3`
- `gemma`

If the model is too large to be run on gpu, it will use some of the available cpu cores. This will be much slower than running entirely on gpu.

### OpenAI
You can use OpenAI LLMs by providing an API key. It depends on the model you want to use, how much it will cost.
You must set the environment variable `OPENAI_API_KEY` with your key.


## Configuration
You must fill a configuration file with the following structure:
```yaml
eureka:
    backend: 'openai'  # 'ollama' or 'openai'
    model: 'gpt-4o'
    temperature: 1.0  # if this value is too low, it is almost deterministic
    iterations: 5
    samples: 8
    use_initial_reward_prompt: false  # if available, use the initial reward prompt

environment:
    name: 'bipedal_walker'
    max_episode_steps: 1600
    class_name: 'BipedalWalker'
    kwargs: null
    benchmark: 'BipedalWalker-v3'  # if benchmark available, set it to train the agent with the same params

experiment:
    parent: 'experiments'
    name: 'bipedal_walker_gpt4o'
    use_datetime: true

rl:
    algo: 'ppo'
    algo_params:
        policy: 'MlpPolicy'
        learning_rate: 0.0003
        n_steps: 2048
        batch_size: 64
        n_epochs: 10
        gamma: 0.999
        gae_lambda: 0.95
        clip_range: 0.2
        ent_coef: 0.0
        vf_coef: 0.5
        max_grad_norm: 0.5

    architecture:
        net_arch: {'pi': [64, 64], 'vf': [64, 64]}
        activation_fn: 'ReLU'
        share_features_extractor: false

    training:
        seed: 0
        eval:
            seed: 5
            num_episodes: 2
            num_evals: 10
        total_timesteps: 1_000_000
        device: 'cuda'
        num_envs: 4
        state_stack: 1
        is_atari: false

    evaluation:
        seed: 10
        num_episodes: 10
        save_gif: true
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
        initial_reward_prompt.txt
```

The code will copy the code into the experiments folder and append the reward function to it. The reward function should 
satisfy the signature:
```python
reward, individual_reward = self.compute_reward(param1, param2, param3)
```
By doing so, the code will be automatically executed by the experiment runner once the reward function is appended.

You must also implement the `self.compute_fitness_score`, the ground truth reward function that allows to compare between 
environments with different reward functions. You can see several implementations on the environments folder:
```python
fitness_score = self.compute_fitness_score(param1, param2, param3)
```

> [!NOTE] 
> The `compute_fitness_score` returns a part of the total fitness score, which is actually the sum over all the episode. 
> Same as reward is the sum of the intermediate rewards during the episode. If the total fitness score is a binary value such as 1 for success, 
> then you will provide always 0 until the episode ends where it will return a 1.

Finally, you must set in the individual_rewards dict the `fitness_score` value:
```python
individual_rewards.update({'fitness_score': fitness_score})
```
This allows us to save all this values for later reward reflection.

> [!TIP]
> You can add a `initial_reward_prompt.txt` with a reward prompt that will be used as the initial reward function (e.g. human-designed reward).

## Contributors

**Rodrigo Sánchez Molina**
  - Email: rsanchezm98@gmail.com
  - Linkedin: [rsanchezm98](https://www.linkedin.com/in/rsanchezm98/)
  - Github: [rsanchezmo](https://github.com/rsanchezmo)

## Citation
If you find `stable-eureka` useful, please consider citing:

```bibtex
  @misc{2024stableeureka,
    title     = {Stable Eureka},
    author    = {Rodrigo Sánchez Molina},
    year      = {2024},
    howpublished = {https://github.com/rsanchezmo/stable-eureka}
  }
```