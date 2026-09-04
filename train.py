"""
Train the pseudo-Siamese ResNet-18 multi-task model on real xBD samples.

Enforces the per-class (12 disaster+severity combos) minimum-sample floor
*before* training, not after: by default it prints a clear warning and
proceeds in "coverage-limited" mode when the floor isn't met (so the
pipeline stays runnable/demonstrable on the currently small real dataset);
pass --strict to make an unmet floor a hard error, per the project rule of
never padding with fake data.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import config
from dataset import XBDDamageDataset, load_metadata, save_splits, stratified_split
from model import build_model
from preprocessing import coverage_report, print_coverage


def check_floor(df, strict):
    report = coverage_report(df)
    print_coverage(report)
    if not report["all_meet_floor"]:
        short = [c for c, v in report["classes"].items() if not v["meets_floor"]]
        msg = (f"{len(short)}/12 disaster+severity classes are below the "
               f"{config.MIN_SAMPLES_FLOOR}-sample floor: {short}. "
               f"See DATA_REPORT.md for sources checked.")
        if strict:
            raise SystemExit(f"[train.py --strict] Refusing to train: {msg}")
        print(f"[train.py] WARNING: {msg}\nProceeding in coverage-limited mode "
              f"(pass --strict to instead abort here).")
    return report


def run_epoch(model, loader, criterion_d, criterion_s, optimizer, device, train):
    model.train(mode=train)
    total_loss, n = 0.0, 0
    for batch in loader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        d_label = batch["disaster_label"].to(device)
        s_label = batch["severity_label"].to(device)

        with torch.set_grad_enabled(train):
            d_logits, s_logits = model(pre, post)
            loss = criterion_d(d_logits, d_label) + criterion_s(s_logits, s_label)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * pre.size(0)
        n += pre.size(0)
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--strict", action="store_true",
                         help="Abort if any of the 12 classes is below the sample floor.")
    parser.add_argument("--no-pretrained", action="store_true",
                         help="Skip attempting to download ImageNet-pretrained weights.")
    args = parser.parse_args()

    df = load_metadata()
    check_floor(df, args.strict)

    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df)
    print(f"\nSplit sizes -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = XBDDamageDataset(train_df, train=True)
    val_ds = XBDDamageDataset(val_df, train=False)
    batch_size = max(1, min(args.batch_size, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, min(args.batch_size, len(val_ds)))) if len(val_ds) else None

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion_d = torch.nn.CrossEntropyLoss()
    criterion_s = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion_d, criterion_s, optimizer, device, train=True)
        val_loss = (run_epoch(model, val_loader, criterion_d, criterion_s, optimizer, device, train=False)
                    if val_loader is not None else float("nan"))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "disaster_types": config.DISASTER_TYPES,
        "severity_levels": config.SEVERITY_LEVELS,
        "image_size": config.IMAGE_SIZE,
        "history": history,
    }, config.CHECKPOINT_PATH)
    print(f"Saved checkpoint to {config.CHECKPOINT_PATH}")

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.epochs + 1), history["train_loss"], label="train loss")
    if val_loader is not None:
        plt.plot(range(1, args.epochs + 1), history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Combined CE loss (disaster + severity)")
    plt.title("Training / validation loss")
    plt.legend()
    plt.tight_layout()
    loss_curve_path = os.path.join(config.FIGURES_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=120)
    plt.close()
    print(f"Saved loss curve to {loss_curve_path}")

    os.makedirs(config.METRICS_DIR, exist_ok=True)
    with open(os.path.join(config.METRICS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
