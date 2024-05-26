from pathlib import Path
import yaml
from datetime import datetime
import ollama
import os

from stable_eureka.logger import get_logger, EmptyLogger
from stable_eureka.utils import read_from_file


class StableEureka:
    def __init__(self, config_path: str):
        if not Path(config_path).exists():
            raise ValueError(f'Config file {config_path} not found')

        self._config = yaml.safe_load(open(config_path, 'r'))

        self._root_path = Path(os.getcwd())
        self._experiment_path = self._root_path / self._config['experiment']['parent'] / self._config['experiment']['name']

        if self._config['experiment']['use_datetime']:
            self._experiment_path /= datetime.utcnow().strftime('%Y-%m-%d')

        self._experiment_path.mkdir(parents=True, exist_ok=True)

        self._initial_system_prompt = read_from_file(self._root_path / 'stable_eureka' / 'prompts' / 'initial_system_prompt.txt')

    def run(self, verbose: bool = True):

        if verbose:
            logger = get_logger()
        else:
            logger = EmptyLogger()

        logger.info(f"Starting stable-eureka optimization. Iterations: {self._config['eureka']['iterations']}, "
                    f"samples: {self._config['eureka']['samples']}")
        logger.info(f"Using LLM: {self._config['eureka']['model']} with T={self._config['eureka']['temperature']}")

        ollama_options = ollama.Options(
            temperature=self._config['eureka']['temperature'],
        )

        for iteration in range(self._config['eureka']['iterations']):
            ...


