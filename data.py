"""CIFAR-10 / CIFAR-100 loaders -- the standard protocol, identical for every method.

Augmentation and normalisation are the recipe every ResNet/WRN CIFAR number is
reported under (Zagoruyko & Komodakis 2016, sec. 4):

    train   RandomCrop(32, padding=4, reflect) -> RandomHorizontalFlip -> Normalize
    test    Normalize

No cutout, no auto-augment, no mixup: the point of this benchmark is the
OPTIMIZER, so every run sees exactly the same data pipeline and the only thing
that varies is `--optimizer` and its hyperparameters.

`--val-frac` carves a validation split out of the 50k training images (by a
FIXED permutation, seed 0, so every run and every method sees the same split).
The tuning stage of `run_ablation.py` selects on that split; the report stage
reads the test split.  With `--val-frac 0` training uses all 50k and the test
set is the only evaluation -- which is what the final runs do.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATASETS = ("cifar10", "cifar100")
N_CLASSES = {"cifar10": 10, "cifar100": 100}
STATS = {                                        # per-channel mean / std
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
DEFAULT_ROOT = os.environ.get("CIFAR_ROOT",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "data"))


def _tfms(name: str, train: bool):
    mean, std = STATS[name]
    norm = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if not train:
        return transforms.Compose(norm)
    return transforms.Compose([transforms.RandomCrop(32, padding=4,
                                                     padding_mode="reflect"),
                               transforms.RandomHorizontalFlip()] + norm)


def _raw(name: str, root: str, train: bool, download: bool):
    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    return cls(root=root, train=train, download=download,
               transform=_tfms(name, train))


def make_loaders(name: str, *, root: Optional[str] = None, batch_size: int = 128,
                 eval_batch_size: int = 512, workers: int = 4, seed: int = 1,
                 val_frac: float = 0.0, download: bool = True,
                 pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, int]:
    """(train_loader, eval_loader, n_classes).

    eval_loader is the validation split when `val_frac > 0`, else the test set.
    The train/val split is deterministic (seed 0) and INDEPENDENT of `seed`, so
    the tuning stage cannot leak a different split into different runs; `seed`
    only drives the shuffling order of the training loader.
    """
    if name not in DATASETS:
        raise KeyError("unknown dataset %r; have %s" % (name, list(DATASETS)))
    root = root or DEFAULT_ROOT
    os.makedirs(root, exist_ok=True)

    train_set = _raw(name, root, True, download)
    if val_frac > 0:
        # val images must NOT be augmented: a second, clean copy of the same files
        clean = _raw(name, root, True, False)
        clean.transform = _tfms(name, False)
        perm = np.random.RandomState(0).permutation(len(train_set))
        n_val = int(round(val_frac * len(train_set)))
        eval_set = Subset(clean, perm[:n_val].tolist())
        train_set = Subset(train_set, perm[n_val:].tolist())
    else:
        eval_set = _raw(name, root, False, download)

    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                    num_workers=workers, pin_memory=pin_memory, drop_last=True,
                    generator=g, persistent_workers=workers > 0)
    el = DataLoader(eval_set, batch_size=eval_batch_size, shuffle=False,
                    num_workers=max(1, workers // 2), pin_memory=pin_memory,
                    persistent_workers=workers > 0)
    return tl, el, N_CLASSES[name]


def download_all(root: Optional[str] = None) -> None:
    """Fetch both datasets once, before launching parallel runs."""
    root = root or DEFAULT_ROOT
    for name in DATASETS:
        _raw(name, root, True, True)
        _raw(name, root, False, True)
    print("CIFAR-10 + CIFAR-100 ready in %s" % root)


if __name__ == "__main__":
    download_all()
