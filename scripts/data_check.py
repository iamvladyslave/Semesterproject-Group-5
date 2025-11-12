from pathlib import Path
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import GTSRBConceptDataset

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg  = load_cfg("config/data_config.yaml")
    dcfg = cfg["dataset"]
    lcfg = cfg["dataloader"]

    tf = transforms.Compose([
        transforms.Resize((dcfg["image_size"], dcfg["image_size"])),
        transforms.ToTensor(),
    ])

    ds = GTSRBConceptDataset(
        root_dir=dcfg["root_dir"],
        concepts_csv=dcfg["concepts_csv"],
        transform=tf,
    )
    print("Using concepts CSV:", dcfg["concepts_csv"])
    print("Train samples:", len(ds), "| num_concepts:", ds.num_concepts)

    ld = DataLoader(
        ds,
        batch_size=lcfg["batch_size"],
        shuffle=lcfg["shuffle"],
        num_workers=lcfg["num_workers"],
        pin_memory=lcfg["pin_memory"],
    )
    xb, (cb, yb) = next(iter(ld))
    print("Batch:", xb.shape, cb.shape, yb.shape)

if __name__ == "__main__":
    main()