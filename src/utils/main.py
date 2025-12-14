import yaml
from src.utils.config import DatasetConfig

with open("config/data_config.yaml", "r") as file:
    config_dict=yaml.safe_load(file)

dataset_config = DatasetConfig(**config_dict["dataset"])

print(dataset_config)