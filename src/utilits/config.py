from dataclasses import dataclass
from typing import Dict

@dataclass 
class DatasetConfig:
    train_dir: str
    test_dir: str
    concept_annotations: str
    batch_size: int
    num_workers: int
    validation_split: float
    
