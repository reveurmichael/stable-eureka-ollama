from stable_eureka import MyStableEurekaForCheckingEnvPy


if __name__ == '__main__':
    m_checker = MyStableEurekaForCheckingEnvPy(config_path='./configs/bipedal_walker_ollama.yml', experiment_datetime='2024-08-20-07-57', iteration=0, sample=2)
    m_checker.run()
