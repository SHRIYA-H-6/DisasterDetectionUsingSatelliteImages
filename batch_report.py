"""
Run inference over every real sample in data/metadata.csv (or a supplied
CSV with the same pre_image_path/post_image_path columns) and write a
per-sample prediction report plus a predicted-class distribution summary.

Usage:
    python batch_report.py [--input-csv data/metadata.csv] [--out outputs/metrics/batch_predictions.csv]
"""
import argparse
import os

import pandas as pd
import torch

import config
from evaluate import load_checkpoint
from predict import predict_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=config.METADATA_CSV)
    parser.add_argument("--out", default=os.path.join(config.METRICS_DIR, "batch_predictions.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(device=device)

    rows = []
    for _, r in df.iterrows():
        pre_path = os.path.join(config.ROOT_DIR, r["pre_image_path"])
        post_path = os.path.join(config.ROOT_DIR, r["post_image_path"])
        result = predict_pair(model, pre_path, post_path, device)
        rows.append({
            "sample_id": r["sample_id"],
            "true_disaster_type": r.get("disaster_type"),
            "true_severity": r.get("severity"),
            "pred_disaster_type": result["disaster_type"],
            "pred_severity": result["severity"],
            "disaster_confidence": max(result["disaster_probs"].values()),
            "severity_confidence": max(result["severity_probs"].values()),
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} predictions to {args.out}")

    print("\nPredicted disaster type distribution:")
    print(out_df["pred_disaster_type"].value_counts().to_string())
    print("\nPredicted severity distribution:")
    print(out_df["pred_severity"].value_counts().to_string())

    if "true_disaster_type" in out_df and out_df["true_disaster_type"].notna().any():
        acc_d = (out_df["pred_disaster_type"] == out_df["true_disaster_type"]).mean()
        acc_s = (out_df["pred_severity"] == out_df["true_severity"]).mean()
        print(f"\nDisaster-type accuracy on this set: {acc_d:.10f}")
        print(f"Severity accuracy on this set:      {acc_s:.10f}")


if __name__ == "__main__":
    main()
