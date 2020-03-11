import torch
from torch.utils.data import DataLoader

from . import datasets as D


def build_dataset(cfg):
    if cfg.DATASET.SAMPLING == "RANDOM_SAMPLE":
        dataset = D.RandomSample(cfg.DATASET)
    elif cfg.DATASET.SAMPLING == "POPULA":
        dataset = D.PopulaSample(cfg.DATASET)
    else:
        raise NotImplementedError()

    return dataset


def build_dataloader(cfg):
    dataset = build_dataset(cfg)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    return data_loader
