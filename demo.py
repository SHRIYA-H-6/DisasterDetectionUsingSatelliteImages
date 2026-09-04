"""
Demo: run the trained model on a handful of real xBD sample images and save
a figure showing each pre/post pair with its true label vs the model's
predicted disaster type + severity.

Usage:
    python demo.py [--n 4]
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

import config
from evaluate import load_checkpoint
from predict import predict_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out", default=os.path.join(config.SAMPLES_DIR, "demo_predictions.png"))
    args = parser.parse_args()

    df = pd.read_csv(config.METADATA_CSV).head(args.n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(device=device)

    rows = len(df)
    fig, axes = plt.subplots(rows, 2, figsize=(7, 3.2 * rows))
    if rows == 1:
        axes = axes.reshape(1, 2)

    for i, (_, r) in enumerate(df.iterrows()):
        pre_path = os.path.join(config.ROOT_DIR, r["pre_image_path"])
        post_path = os.path.join(config.ROOT_DIR, r["post_image_path"])
        result = predict_pair(model, pre_path, post_path, device)

        print(f"{r['sample_id']}: true={r['disaster_type']}_{r['severity']} "
              f"pred={result['disaster_type']}_{result['severity']} "
              f"(disaster p={max(result['disaster_probs'].values()):.10f}, "
              f"severity p={max(result['severity_probs'].values()):.10f})")

        for j, (path, tag) in enumerate([(pre_path, "PRE"), (post_path, "POST")]):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(f"{r['sample_id']} {tag}\ntrue={r['disaster_type']}_{r['severity']}", fontsize=8)
            else:
                axes[i, j].set_title(f"{tag}\npred={result['disaster_type']}_{result['severity']}", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"\nSaved demo figure to {args.out}")


if __name__ == "__main__":
    main()
