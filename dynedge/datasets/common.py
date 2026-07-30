"""ImageFolder-style loaders for prepared FERPlus / ExpW caches."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class GrayTo3Ch:
    """MCU 方案 A: RGB/灰度 → 单通道灰度 → 复制为 3 通道（共享 stem）。"""

    def __call__(self, img):
        g = img.convert("L")
        return g.convert("RGB")  # L→RGB copies gray into R=G=B


def build_transforms(
    train: bool,
    img_size: int = 112,
    aug: str = "strong",
    gray_to_3ch: bool = False,
):
    """
    aug:
      - none: flip only
      - mild: flip + light crop (good for FERPlus 48→112 upsampled faces)
      - strong: RandAugment (ExpW outdoor)
    gray_to_3ch:
      - True: force grayscale then replicate to 3ch (MCU 64×64 distribution)
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    head: list = []
    if gray_to_3ch:
        head.append(GrayTo3Ch())

    if not train:
        return transforms.Compose(
            head
            + [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                norm,
            ]
        )

    ops: list = head + [transforms.RandomHorizontalFlip()]
    if aug == "mild":
        ops.append(transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)))
        ops.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))
    elif aug == "strong":
        ops.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
        ops.append(transforms.RandAugment(num_ops=2, magnitude=7))
    else:  # none
        ops.append(transforms.Resize((img_size, img_size)))
    ops.extend([transforms.ToTensor(), norm])
    return transforms.Compose(ops)


def make_loaders(
    root: Path,
    batch_size: int = 1024,
    num_workers: int = 16,
    img_size: int = 112,
    pin_memory: bool = True,
    aug: str = "strong",
    gray_to_3ch: bool = False,
):
    """root contains train/val/test class subdirs."""
    train_ds = datasets.ImageFolder(
        root / "train",
        transform=build_transforms(True, img_size, aug=aug, gray_to_3ch=gray_to_3ch),
    )
    val_ds = datasets.ImageFolder(
        root / "val",
        transform=build_transforms(False, img_size, gray_to_3ch=gray_to_3ch),
    )
    test_ds = datasets.ImageFolder(
        root / "test",
        transform=build_transforms(False, img_size, gray_to_3ch=gray_to_3ch),
    )

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=shuffle,
        )

    return {
        "train": _loader(train_ds, True),
        "val": _loader(val_ds, False),
        "test": _loader(test_ds, False),
        "num_classes": len(train_ds.classes),
        "classes": train_ds.classes,
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "test_len": len(test_ds),
        "img_size": img_size,
        "gray_to_3ch": gray_to_3ch,
    }
