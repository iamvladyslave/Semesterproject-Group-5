from dataclasses import dataclass
from typing import Dict

@dataclass
class DatasetConfig:
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
