from stable_eureka import StableEureka


if __name__ == '__main__':
    trainer = StableEureka(config_path='./configs/bipedal_walker.yml')
    trainer.run(verbose=True)