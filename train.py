from stable_eureka import StableEureka


if __name__ == '__main__':
    trainer = StableEureka(config_path='./configs/mountain_car_continuous.yml')
    trainer.run(verbose=True)
