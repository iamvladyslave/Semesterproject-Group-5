from dataclasses import dataclass
from typing import Dict

@dataclass
class DatasetConfig:
    '''
    configuration for dataset loading
    name: str
        name of dataset
    root_dir: str
        root directory containing data
    test_root_dir: str
        root directory for test data
    train_csv: str
        optional path to csv file contataining annotations for training
    val_csv: str
        optional path to csv file contataining annotations for validation
    test_csv: str
        path to csv file contataining annotations used for testing
    concepts_csv: str
        path to csv file containing concept annotations
    image_exts: str tuple
        allowed image file extensions
    image_size: int
        size images are being resized to
    val_split: float
        which amount of data is split into validation set
    test_split
        which amount of data is split int test set
    seed: int
        random seed for dataset splitting
    num_concepts: int
        number of concepts in dataset
    concept_cols: str list
        names of columns in csv corresponding to concept labels
    folder_to_label: int dict
        mapping from folder indices to class labels

    
    '''
    name: str
    root_dir: str
    test_root_dir: str | None = None
    train_csv: str | None = None
    val_csv: str | None = None
    test_csv: str | None = None
    concepts_csv: str
    image_exts: tuple[str, ...] = (".ppm",)
    image_size: int = 128
    val_split: float = 0.1
    test_split: float = 0.0
    seed: int = 1923
    num_concepts: int = 43
    concept_cols: list[str] | None = None
    folder_to_label: dict[int, int] | None = None
