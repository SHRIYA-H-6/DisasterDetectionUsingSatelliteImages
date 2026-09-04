"""
PyTorch Dataset for xBD pre/post disaster image pairs, plus a stratified
train/val/test splitter keyed on the joint disaster_type+severity label.
"""
import os
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

import config


def load_metadata(csv_path=config.METADATA_CSV):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python preprocessing.py` first."
        )
    return pd.read_csv(csv_path)


def _can_stratify(labels, n_splits):
    """sklearn requires every class to have at least n_splits members to
    stratify a split into n_splits parts."""
    counts = pd.Series(labels).value_counts()
    return len(counts) > 1 and counts.min() >= n_splits


def stratified_split(df, val_size=0.15, test_size=0.15, seed=config.RANDOM_SEED):
    """Split df into train/val/test, stratified on `joint_label` when every
    class has enough members to support it; otherwise falls back to a plain
    random split and prints a warning (this is expected while the real
    dataset is still far below the per-class floor -- see DATA_REPORT.md)."""
    labels = df["joint_label"].values

    if _can_stratify(labels, 2):
        train_df, temp_df = train_test_split(
            df, test_size=val_size + test_size, stratify=labels, random_state=seed
        )
        temp_labels = temp_df["joint_label"].values
        if _can_stratify(temp_labels, 2) and len(temp_df) >= 2:
            rel_test = test_size / (val_size + test_size)
            val_df, test_df = train_test_split(
                temp_df, test_size=rel_test, stratify=temp_labels, random_state=seed
            )
        else:
            warnings.warn(
                "Not enough samples per class to stratify the val/test split; "
                "falling back to a random split for that portion."
            )
            rel_test = test_size / (val_size + test_size)
            val_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=seed)
    else:
        warnings.warn(
            "Not enough samples per joint class to stratify train/val/test "
            f"(need >=2 per class; dataset has {len(df)} rows across "
            f"{df['joint_label'].nunique()} classes). Falling back to a plain "
            "random split. This is expected until real per-class coverage "
            "reaches the floor -- see DATA_REPORT.md."
        )
        n = len(df)
        if n < 3:
            # Too small to split at all -- everything goes to train, and
            # val/test are copies of train so downstream code still runs.
            return df.copy(), df.copy(), df.copy()
        train_df, temp_df = train_test_split(df, test_size=val_size + test_size, random_state=seed)
        rel_test = test_size / (val_size + test_size)
        val_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=seed)

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def save_splits(train_df, val_df, test_df, out_dir=config.SPLITS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)


def get_transform(train):
    ops = [
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class XBDDamageDataset(Dataset):
    """Loads real pre/post satellite image pairs and their disaster_type /
    severity labels, derived from real xBD damage annotations."""

    def __init__(self, df, train=False, root_dir=config.ROOT_DIR):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = get_transform(train)
        self.disaster_to_idx = {d: i for i, d in enumerate(config.DISASTER_TYPES)}
        self.severity_to_idx = {s: i for i, s in enumerate(config.SEVERITY_LEVELS)}

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        img = cv2.imread(os.path.join(self.root_dir, rel_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {rel_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pre_img = self._load_image(row["pre_image_path"])
        post_img = self._load_image(row["post_image_path"])
        pre_t = self.transform(pre_img)
        post_t = self.transform(post_img)
        disaster_idx = self.disaster_to_idx[row["disaster_type"]]
        severity_idx = self.severity_to_idx[row["severity"]]
        return {
            "pre": pre_t,
            "post": post_t,
            "disaster_label": torch.tensor(disaster_idx, dtype=torch.long),
            "severity_label": torch.tensor(severity_idx, dtype=torch.long),
            "sample_id": row["sample_id"],
            "disaster_type": row["disaster_type"],
            "severity": row["severity"],
        }


if __name__ == "__main__":
    df = load_metadata()
    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    ds = XBDDamageDataset(train_df, train=True)
    if len(ds) > 0:
        sample = ds[0]
        print({k: (v.shape if isinstance(v, torch.Tensor) else v) for k, v in sample.items()})
