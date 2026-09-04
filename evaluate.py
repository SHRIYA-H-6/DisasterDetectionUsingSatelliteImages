"""
Evaluate the trained model on the held-out test split and produce the three
required report files (disaster type alone, severity alone, and the 12-way
joint disaster+severity report), each with precision/recall/f1 rounded to 10
decimal places, plus 4x4 / 3x3 / 12x12 confusion matrix heatmaps.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

import config
from dataset import XBDDamageDataset, load_metadata, stratified_split
from model import build_model


def load_checkpoint(path=config.CHECKPOINT_PATH, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python train.py` first.")
    ckpt = torch.load(path, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_inference(model, loader, device):
    true_d, pred_d, true_s, pred_s, sample_ids = [], [], [], [], []
    for batch in loader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        d_logits, s_logits = model(pre, post)
        pred_d.extend(d_logits.argmax(dim=1).cpu().tolist())
        pred_s.extend(s_logits.argmax(dim=1).cpu().tolist())
        true_d.extend(batch["disaster_label"].tolist())
        true_s.extend(batch["severity_label"].tolist())
        sample_ids.extend(batch["sample_id"])
    return true_d, pred_d, true_s, pred_s, sample_ids


def _round_report(report_dict, decimals=10):
    """Round every precision/recall/f1-score float in a
    classification_report(output_dict=True) dict to `decimals` places, and
    cast support to int. The scalar 'accuracy' entry is rounded directly."""
    out = {}
    for key, val in report_dict.items():
        if key == "accuracy":
            out[key] = round(float(val), decimals)
        elif isinstance(val, dict):
            out[key] = {
                metric: (int(v) if metric == "support" else round(float(v), decimals))
                for metric, v in val.items()
            }
        else:
            out[key] = val
    return out


def build_reports(true_d, pred_d, true_s, pred_s):
    disaster_names = config.DISASTER_TYPES
    severity_names = config.SEVERITY_LEVELS
    true_d_names = [disaster_names[i] for i in true_d]
    pred_d_names = [disaster_names[i] for i in pred_d]
    true_s_names = [severity_names[i] for i in true_s]
    pred_s_names = [severity_names[i] for i in pred_s]
    true_joint = [f"{d}_{s}" for d, s in zip(true_d_names, true_s_names)]
    pred_joint = [f"{d}_{s}" for d, s in zip(pred_d_names, pred_s_names)]

    disaster_report = classification_report(
        true_d_names, pred_d_names, labels=disaster_names,
        output_dict=True, zero_division=0,
    )
    severity_report = classification_report(
        true_s_names, pred_s_names, labels=severity_names,
        output_dict=True, zero_division=0,
    )
    combined_report = classification_report(
        true_joint, pred_joint, labels=config.JOINT_CLASSES,
        output_dict=True, zero_division=0,
    )
    return (_round_report(disaster_report), _round_report(severity_report),
            _round_report(combined_report), true_joint, pred_joint,
            true_d_names, pred_d_names, true_s_names, pred_s_names)


def save_reports(disaster_report, severity_report, combined_report):
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    paths = {
        "disaster_type_report.json": disaster_report,
        "severity_report.json": severity_report,
        "combined_report.json": combined_report,
    }
    for name, report in paths.items():
        p = os.path.join(config.METRICS_DIR, name)
        with open(p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {p}")


def print_combined_console(combined_report):
    print("\n=== Disaster + Severity combined report (console) ===")
    for disaster in config.DISASTER_TYPES:
        for severity in config.SEVERITY_LEVELS:
            cls = f"{disaster}_{severity}"
            entry = combined_report.get(cls, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
            print(f"{disaster} {severity}:")
            print(f"  Accuracy (F1-score): {entry['f1-score']:.10f}")
            print(f"  Precision: {entry['precision']:.10f}")
            print(f"  Recall: {entry['recall']:.10f}")
            print(f"  Support: {entry['support']}")
    print(f"\nOverall joint accuracy: {combined_report['accuracy']:.10f}")
    print(f"Macro avg   -> precision={combined_report['macro avg']['precision']:.10f} "
          f"recall={combined_report['macro avg']['recall']:.10f} "
          f"f1={combined_report['macro avg']['f1-score']:.10f}")
    print(f"Weighted avg-> precision={combined_report['weighted avg']['precision']:.10f} "
          f"recall={combined_report['weighted avg']['recall']:.10f} "
          f"f1={combined_report['weighted avg']['f1-score']:.10f}")


def print_simple_console(name, report, class_names):
    print(f"\n=== {name} report (console) ===")
    for cls in class_names:
        entry = report.get(cls, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        print(f"{cls}: precision={entry['precision']:.10f} recall={entry['recall']:.10f} "
              f"f1={entry['f1-score']:.10f} support={entry['support']}")
    print(f"Accuracy: {report['accuracy']:.10f}")


def save_confusion_matrix(true_labels, pred_labels, class_names, title, out_path):
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    size = max(6, len(class_names) * 0.9)
    plt.figure(figsize=(size, size * 0.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(device=device)

    test_csv = os.path.join(config.SPLITS_DIR, "test.csv")
    if os.path.exists(test_csv):
        import pandas as pd
        test_df = pd.read_csv(test_csv)
    else:
        df = load_metadata()
        _, _, test_df = stratified_split(df)

    test_ds = XBDDamageDataset(test_df, train=False)
    test_loader = DataLoader(test_ds, batch_size=max(1, min(args.batch_size, len(test_ds))))

    true_d, pred_d, true_s, pred_s, sample_ids = run_inference(model, test_loader, device)

    (disaster_report, severity_report, combined_report, true_joint, pred_joint,
     true_d_names, pred_d_names, true_s_names, pred_s_names) = build_reports(true_d, pred_d, true_s, pred_s)

    save_reports(disaster_report, severity_report, combined_report)

    print_simple_console("Disaster type", disaster_report, config.DISASTER_TYPES)
    print_simple_console("Severity", severity_report, config.SEVERITY_LEVELS)
    print_combined_console(combined_report)

    save_confusion_matrix(true_d_names, pred_d_names, config.DISASTER_TYPES,
                           "Disaster type confusion matrix (4x4)",
                           os.path.join(config.FIGURES_DIR, "confusion_matrix_disaster_type.png"))
    save_confusion_matrix(true_s_names, pred_s_names, config.SEVERITY_LEVELS,
                           "Severity confusion matrix (3x3)",
                           os.path.join(config.FIGURES_DIR, "confusion_matrix_severity.png"))
    save_confusion_matrix(true_joint, pred_joint, config.JOINT_CLASSES,
                           "Disaster+Severity combined confusion matrix (12x12)",
                           os.path.join(config.FIGURES_DIR, "confusion_matrix_combined.png"))

    print(f"\nEvaluated on {len(test_ds)} real test sample(s): {sample_ids}")
    print("NOTE: with the currently small real-data test set this evaluation demonstrates "
          "the metric pipeline's correctness, not a claim of final model performance -- "
          "see DATA_REPORT.md.")


if __name__ == "__main__":
    main()
