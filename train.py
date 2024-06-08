from stable_eureka import StableEureka


if __name__ == '__main__':
    trainer = StableEureka(config_path='./configs/lunar_lander.yml')
    trainer.run(verbose=True)
