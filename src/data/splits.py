from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.preprocessing import Preprocessing
from src.dataset import GTSRBConceptCSVDataset, GTSRBConceptDataset


def _make_transform(dataset_cfg: dict):
    '''
    transforms images using preprocessing.py based on imagesize set in dataset
    Parameters
    ----------
    dataset_cfg: dict
        dataset config dictonary containing "image_size"

    Returns
    -------
        transformed images on which preprocessing has been applied

    Examples
    --------
    >>>
    '''
    image_size = dataset_cfg["image_size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    return Preprocessing(image_size=image_size).transform


def _split_indices(
    labels: list[int],
    *,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[list[int], list[int], list[int]]:
    '''
    splits dataset indices into train-, val- and testsplit

    Parameters
    ----------
    labels: int list
        class labels for samples from dataset
    val_split: float
        amount of data used for validationset
    test_split: float
        amount of data used for testset
    seed: int
        fixed random seed for reproducibility 

    Returns
    -------
    train_idx: int list
        indices for training samples
    val_idx: int list
        indices for validation samples
    test_idx: int list
        indices for test samples
    Examples
    --------
    >>>
    '''
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    indices = list(range(len(labels)))
    train_idx, val_idx, test_idx = indices, [], []

    if test_split > 0:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_split,
            random_state=seed,
            stratify=labels,
        )

    if val_split > 0:
        train_labels = [labels[i] for i in train_idx]
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_split / max(1e-8, (1 - test_split)),
            random_state=seed,
            stratify=train_labels,
        )

    return list(train_idx), list(val_idx), list(test_idx)


def build_datasets(dataset_cfg: dict) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    '''
    build train, val und testsets with annotations
    
    parameters
    ----------
    dataset_cfg: dict
        dataset config dictonary containing path, splitamount and settings
    
    Returns
    -------
    train_ds: dataset
        is the training dataset
    val_ds: dataset or none
        is the validation dataset if available
    test_ds: dataset or none
        is the test dataset if available
    
    Examples
    --------
    >>>
    '''
    transform = _make_transform(dataset_cfg)
    image_exts = tuple(dataset_cfg.get("image_exts", [".ppm"]))
    concepts_csv = dataset_cfg["concepts_csv"]
    seed = dataset_cfg.get("seed", 1923)

    if dataset_cfg.get("train_csv") or dataset_cfg.get("val_csv"):
        raise ValueError(
            "Train/val CSV splits are not supported for Tasks 1-3. "
            "Use root_dir with val_split/test_split instead."
        )

    test_ds = None
    test_csv = dataset_cfg.get("test_csv")
    if test_csv:
        test_root = dataset_cfg.get("test_root_dir", dataset_cfg["root_dir"])
        test_ds = GTSRBConceptCSVDataset(
            root_dir=test_root,
            csv_path=test_csv,
            concepts_csv=concepts_csv,
            transform=transform,
            image_exts=image_exts,
            require_labels=False,
        )

    base_ds = GTSRBConceptDataset(
        root_dir=dataset_cfg["root_dir"],
        concepts_csv=concepts_csv,
        transform=transform,
        image_exts=image_exts,
    )
    labels = [lbl for _, lbl in base_ds.samples]
    val_split = dataset_cfg.get("val_split", 0.1)
    test_split = 0.0 if test_ds is not None else dataset_cfg.get("test_split", 0.0)
    train_idx, val_idx, test_idx = _split_indices(
        labels,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    train_ds = Subset(base_ds, train_idx)
    val_ds = Subset(base_ds, val_idx) if val_idx else None
    if test_ds is None and test_idx:
        test_ds = Subset(base_ds, test_idx)
    return train_ds, val_ds, test_ds


def _seed_worker(worker_id: int, seed: int) -> None:
    '''
    Initializes (fixed) random seed for dataloader
    Parameters
    ----------
    worker_id: int
        ID of worker process
    seed: int
        fixed random seed

    Examples
    -------
    >>>
    '''
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloaders(cfg: dict):
    '''
    Creates Pytorch Dataloader for train, val and test
    Builds datasets, wraps them in dataloaders, applies seeding using fixed random seed

    Parameters
    ----------
    cfg: dict
        configuration dictonary containing dataset and dataloader configurations
    
    Returns
    -------
    train_ds: dataset
        train dataset 

    val_ds:dataset or none
        val dataset if available
    test_ds: dataset or none
        test dataset if available
    train_loader: dataloader
        train dataloader
    val_loader: dataloader or none
        val dataloader if available
    test_loader: dataloader or none
        test dataloader if available
    Examples
    --------
    >>>
    '''
    dataset_cfg = cfg["dataset"]
    dataloader_cfg = cfg["dataloader"]
    seed = dataset_cfg.get("seed", 1923)

    train_ds, val_ds, test_ds = build_datasets(dataset_cfg)
    generator = torch.Generator().manual_seed(seed)

    def make_loader(ds: Optional[Dataset], shuffle: bool):
        '''
        creates a pytorch dataloader for given dataset. Creates batches in given size and
        shuffles them optionally
        
        Parameters
        ----------
        ds: Dataset or None
            Dataset to wrap
        shuffle: bool
            if true then shuffles the dataset during loading

        Returns
        --------
        Dataloader or none
            returns dataloader using given configurations if dataset is provided

        Examples
        ---------
        >>>
        '''
        if ds is None:
            return None
        return DataLoader(
            ds,
            batch_size=dataloader_cfg["batch_size"],
            shuffle=shuffle,
            num_workers=dataloader_cfg["num_workers"],
            pin_memory=dataloader_cfg.get("pin_memory", True),
            worker_init_fn=lambda wid: _seed_worker(wid, seed),
            generator=generator,
        )

    train_loader = make_loader(train_ds, dataloader_cfg.get("shuffle", True))
    val_loader = make_loader(val_ds, False)
    test_loader = make_loader(test_ds, False)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
