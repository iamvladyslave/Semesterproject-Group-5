from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Images

class GTSRBConceptDataset(Dataset):
    def __init__(self, root_dir, csv_file, concepts_csv, transform=None):
        """
        root_dir: train folder path ()
        csv_file: path to GT-final_train.csv
        concepts_csv: path to concept_per_class.csv
        transform: torchvision transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.data = pd.read_csv(csv_file)

        concept_df = pd.read_csv(concepts_csv)

        concept_cols = [c for c in concept_df.columns if c not in ("class_id", "class_name")]
        self.concept_cols = concept_cols

        self.class_to_concept = {
            int(row["class_id"]): torch.tensor(row[concept_cols].values, dtype=torch.float32)
            for _, row in concept_df.iterrows()
        }

        # Sanity check
        missing = set(self.data["ClassId"]) - set(self.class_to_concept.keys())
        if missing:
            raise ValueError(f"Missing concept rows for class IDs: {missing}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        class_id = int(row["ClassId"])
        subdir = f"{class_id:05d}"
        img_path = self.root_dir / subdir / row["Filename"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Lookup concept vector for this class
        concept_vec = self.class_to_concept[class_id]
        label = torch.tensor(class_id, dtype=torch.long)

        return image, (concept_vec, label)
