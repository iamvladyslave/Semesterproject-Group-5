from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
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
        image_exts: Tuple[str, ...] = (".ppm", ".png", ".jpg", ".jpeg", ".bmp"),
        crop_with_roi: bool = True,
        concepts_class_col: str = "class_id",
        concepts_name_col: str = "class_name",
        concept_columns: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_exts = tuple(e.lower() for e in image_exts)
        self.crop_with_roi = crop_with_roi

        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")

        class_dirs = sorted([
            p for p in self.root_dir.rglob("[0-9][0-9][0-9][0-9][0-9]") if p.is_dir()
        ])
        if not class_dirs:
            raise RuntimeError(
                f"No class folders like 00000..00042 under {self.root_dir} (checked recursively)"
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


        self._roi_by_path: Dict[Path, Tuple[int, int, int, int]] = {}
        if self.crop_with_roi:
            for cdir in class_dirs:
                csvs = list(cdir.glob("*.csv"))
                if not csvs:
                    continue
                df = pd.read_csv(csvs[0])
                if {"Filename", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"}.issubset(df.columns):
                    for _, row in df.iterrows():
                        roi_path = cdir / str(row["Filename"])
                        if roi_path.exists():
                            x1, y1, x2, y2 = int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"])
                            self._roi_by_path[roi_path] = (x1, y1, x2, y2)

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
            roi = self._roi_by_path.get(path)
            if roi is not None:
                x1, y1, x2, y2 = roi
                im = im.crop((x1, y1, x2, y2))

            if self.transform is not None:
                image = self.transform(im)
            else:
                image = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        concept_vec = self._concept_by_label[label]
        label_t = torch.tensor(label, dtype=torch.long)
        return image, (concept_vec, label_t)