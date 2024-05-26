from pathlib import Path
import yaml
from datetime import datetime


class StableEureka:
    def __init__(self, config_path: str):
        if not Path(config_path).exists():
            raise ValueError(f'Config file {config_path} not found')

        self.config = yaml.safe_load(open(config_path, 'r'))

        self._experiment_path = Path(self.config['experiment']['parent']) / self.config['experiment']['name']

        if self.config['experiment']['use_datetime']:
            self._experiment_path /= datetime.utcnow().strftime('%Y-%m-%d')

        self._experiment_path.mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = False):
        ...
