from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class GTSRBConceptDataset(Dataset):
    """
    GTSRB + Concept annotations Dataset

    Directory layout expected:
        root_dir/
          00000/  (class_id = 0)
            00000_00000.ppm
            ...
            GT-00000.csv
          00001/
            ...
          ...
          00042/

    Concepts CSV (wide, per class);
        class_id,class_name,has_red_border,is_round,is_triangle,...
        0,Speed limit (20km/h),1,1,0,...
        ...

    Returns:
        (image_tensor, (concept_vector_tensor, label_tensor))

        image_tensor: FloatTensor [C,H,W] after transforms
        concept_vector_tensor: FloatTensor [K] with 0/1 values
        label_tensor: LongTensor scalar (the class_id)
    """

    def __init__(
        self,
        root_dir: str | Path,
        concepts_csv: str | Path,
        *,
        transform=None,
        image_exts: Tuple[str, ...] = (".ppm",),
        concepts_class_col: str = "class_id",
        concepts_name_col: str = "class_name",
        concept_columns: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_exts = tuple(e.lower() for e in image_exts)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")

        class_dirs = sorted([
            p for p in self.root_dir.rglob("[0-9][0-9][0-9][0-9][0-9]") if p.is_dir()
        ])
        if not class_dirs:
            raise RuntimeError(
                f"No class folders under {self.root_dir} (checked recursively)"
            )

        samples: List[Tuple[Path, int]] = []
        for cdir in class_dirs:
            try:
                class_id = int(cdir.name)
            except ValueError:
                continue
            for img_path in cdir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in self.image_exts:
                    samples.append((img_path, class_id))


        if not samples:
            raise RuntimeError(f"No images with extensions {self.image_exts} found under {self.root_dir}")

        self.samples = sorted(samples, key=lambda t: (t[1], t[0].name))
        self.classes = sorted({lbl for _, lbl in self.samples})
        self.class_to_idx = {c: c for c in self.classes}



        cdf = pd.read_csv(concepts_csv)
        if concepts_class_col not in cdf.columns:
            raise KeyError(f"Concepts CSV must have column '{concepts_class_col}'")

        if concept_columns is None:
            drop = {concepts_class_col}
            if concepts_name_col in cdf.columns:
                drop.add(concepts_name_col)
            concept_columns = [c for c in cdf.columns if c not in drop]
        self.concept_columns = list(concept_columns)
        self.num_concepts = len(self.concept_columns)
        if self.num_concepts == 0:
            raise ValueError("No concept columns found. Pass concept_columns=... or check your CSV.")

        self._concept_by_label: Dict[int, torch.Tensor] = {}
        for _, row in cdf.iterrows():
            cid = int(row[concepts_class_col])
            vec = torch.tensor([float(row[c]) for c in self.concept_columns], dtype=torch.float32)
            vec = (vec > 0.5).float()
            self._concept_by_label[cid] = vec

        labels_in_split = {lbl for _, lbl in self.samples}
        missing = sorted([lbl for lbl in labels_in_split if lbl not in self._concept_by_label])
        if missing:
            raise KeyError(f"Concept vectors missing for class_id(s): {missing}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.transform is not None:
                image = self.transform(im)
            else:
                image = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        concept_vec = self._concept_by_label[label]
        label_t = torch.tensor(label, dtype=torch.long)
        return image, (concept_vec, label_t)


def _infer_csv_sep(path: Path) -> str:
    with open(path, "r") as f:
        header = f.readline()
    return ";" if ";" in header else ","


def _find_label_column(columns: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for candidate in ("classid", "class_id", "label", "class"):
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _load_concept_vectors(
    concepts_csv: str | Path,
    *,
    concepts_class_col: str = "class_id",
    concepts_name_col: str = "class_name",
    concept_columns: Optional[List[str]] = None,
):
    cdf = pd.read_csv(concepts_csv)
    if concepts_class_col not in cdf.columns:
        raise KeyError(f"Concepts CSV must have column '{concepts_class_col}'")

    if concept_columns is None:
        drop = {concepts_class_col}
        if concepts_name_col in cdf.columns:
            drop.add(concepts_name_col)
        concept_columns = [c for c in cdf.columns if c not in drop]
    concept_columns = list(concept_columns)
    if not concept_columns:
        raise ValueError("No concept columns found. Pass concept_columns=... or check your CSV.")

    concept_by_label: Dict[int, torch.Tensor] = {}
    for _, row in cdf.iterrows():
        cid = int(row[concepts_class_col])
        vec = torch.tensor([float(row[c]) for c in concept_columns], dtype=torch.float32)
        vec = (vec > 0.5).float()
        concept_by_label[cid] = vec
    return concept_columns, concept_by_label


def _load_concept_columns(
    concepts_csv: str | Path,
    *,
    concepts_class_col: str = "class_id",
    concepts_name_col: str = "class_name",
    concept_columns: Optional[List[str]] = None,
):
    cdf = pd.read_csv(concepts_csv)
    if concept_columns is None:
        drop = {concepts_class_col}
        if concepts_name_col in cdf.columns:
            drop.add(concepts_name_col)
        concept_columns = [c for c in cdf.columns if c not in drop]
    concept_columns = list(concept_columns)
    if not concept_columns:
        raise ValueError("No concept columns found. Pass concept_columns=... or check your CSV.")
    return concept_columns


class GTSRBConceptCSVDataset(Dataset):
    """
    Dataset backed by a CSV file with filenames.
    """

    def __init__(
        self,
        root_dir: str | Path,
        csv_path: str | Path,
        concepts_csv: str | Path,
        *,
        transform=None,
        image_exts: Tuple[str, ...] = (".ppm",),
        concept_columns: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        require_labels: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_exts = tuple(e.lower() for e in image_exts)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        sep = _infer_csv_sep(csv_path)
        df = pd.read_csv(csv_path, sep=sep)
        filename_col = "Filename" if "Filename" in df.columns else "filename"
        if filename_col not in df.columns:
            raise KeyError("CSV must include a Filename column")

        label_col = label_col or _find_label_column(list(df.columns))
        if label_col is None and require_labels:
            raise KeyError(
                "CSV must include a label column (ClassId/class_id/label) "
                "to evaluate labels and concepts."
            )
        self.has_labels = label_col is not None

        self.samples: List[Tuple[Path, int]] = []
        for _, row in df.iterrows():
            filename = str(row[filename_col])
            img_path = self.root_dir / filename
            if not img_path.exists():
                continue
            if img_path.suffix.lower() not in self.image_exts:
                continue
            label = int(row[label_col]) if self.has_labels else -1
            self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No valid images found for CSV {csv_path}")

        self.samples = sorted(self.samples, key=lambda t: t[0].name)
        self.classes = sorted({lbl for _, lbl in self.samples if lbl >= 0})
        self.class_to_idx = {c: c for c in self.classes}

        self.concept_columns, self._concept_by_label = _load_concept_vectors(
            concepts_csv,
            concept_columns=concept_columns,
        )
        self.num_concepts = len(self.concept_columns)
        self.num_classes = len(self._concept_by_label)
        self.classes = sorted(self._concept_by_label.keys())
        self.class_to_idx = {c: c for c in self.classes}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(im)
        else:
            image = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0

        if self.has_labels:
            concept_vec = self._concept_by_label[label]
        else:
            concept_vec = torch.zeros(self.num_concepts, dtype=torch.float32)
        label_t = torch.tensor(label, dtype=torch.long)
        return image, (concept_vec, label_t)
