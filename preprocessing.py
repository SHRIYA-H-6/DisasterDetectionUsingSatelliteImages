"""
Build data/metadata.csv from real xBD (xView2) imagery + official damage-annotation
JSON in data/raw/xbd/, derive tile-level severity from building damage subtypes,
save a sample grid of loaded images for a visual "these are real satellite photos"
check, and report real-sample coverage against the 12 disaster+severity classes.

No image is ever generated or synthesized here: every row in metadata.csv points
at a real PNG that was downloaded/copied from a verified real-data source (see
data/raw/SOURCES.md).
"""
import argparse
import collections
import glob
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


def _pre_post_paths(event_id):
    """event_id, e.g. 'hurricane-florence_00000086' -> (pre_png, post_png) or (None, None)."""
    pre = os.path.join(config.RAW_XBD_IMAGES_DIR, f"{event_id}_pre_disaster.png")
    post = os.path.join(config.RAW_XBD_IMAGES_DIR, f"{event_id}_post_disaster.png")
    return (pre if os.path.exists(pre) else None,
            post if os.path.exists(post) else None)


def derive_severity(post_json_path):
    """Majority-vote tile-level severity from per-building xBD damage subtypes.

    Returns (severity, num_buildings, num_classified) where severity is None
    if there are zero classifiable buildings (nothing to vote on).
    """
    with open(post_json_path) as f:
        label = json.load(f)
    subtypes = [
        feat["properties"].get("subtype")
        for feat in label.get("features", {}).get("xy", [])
    ]
    votes = collections.Counter()
    for st in subtypes:
        tier = config.XBD_SUBTYPE_TO_SEVERITY.get(st)
        if tier is not None:
            votes[tier] += 1
    if not votes:
        return None, len(subtypes), 0
    severity = votes.most_common(1)[0][0]
    return severity, len(subtypes), sum(votes.values())


def scan_xbd():
    """Scan data/raw/xbd/labels for post_disaster.json files and build rows.

    A row requires: a real post_disaster.json label, a real pre_disaster.png,
    a real post_disaster.png, a disaster_type xBD maps to one of our 4 classes,
    and at least one classifiable building (so a severity can be derived).
    """
    rows = []
    skipped = []
    label_files = sorted(glob.glob(os.path.join(config.RAW_XBD_LABELS_DIR, "*_post_disaster.json")))
    for label_path in label_files:
        fname = os.path.basename(label_path)
        event_id = fname[: -len("_post_disaster.json")]

        with open(label_path) as f:
            meta = json.load(f)["metadata"]
        xbd_type = meta.get("disaster_type")
        disaster_type = config.XBD_DISASTER_TYPE_MAP.get(xbd_type)
        if disaster_type is None:
            skipped.append((event_id, f"xBD disaster_type '{xbd_type}' not mapped to a target class"))
            continue

        pre_png, post_png = _pre_post_paths(event_id)
        if pre_png is None or post_png is None:
            skipped.append((event_id, "missing real pre/post image pair (label-only, no pixels available)"))
            continue

        severity, n_bldg, n_classified = derive_severity(label_path)
        if severity is None:
            skipped.append((event_id, f"0 classifiable buildings out of {n_bldg} (cannot derive severity)"))
            continue

        rows.append({
            "sample_id": event_id,
            "disaster_event": meta.get("disaster"),
            "disaster_type": disaster_type,
            "severity": severity,
            "joint_label": f"{disaster_type}_{severity}",
            "pre_image_path": os.path.relpath(pre_png, config.ROOT_DIR),
            "post_image_path": os.path.relpath(post_png, config.ROOT_DIR),
            "num_buildings": n_bldg,
            "num_classified_buildings": n_classified,
            "sensor": meta.get("sensor"),
            "gsd": meta.get("gsd"),
            "capture_date": meta.get("capture_date"),
            "source": "xBD (Gupta et al., 2019)",
        })
    return rows, skipped


def scan_landslide():
    """Scan data/raw/landslide for a labels.csv describing real supplementary
    landslide pre/post pairs + severity (Sentinel-1 SAR or equivalent real
    imagery). Returns [] if nothing has been placed there yet -- this is the
    documented drop-in point for real landslide data (see DATA_REPORT.md).

    Expected data/raw/landslide/labels.csv columns:
      sample_id, pre_image_path, post_image_path, severity, source
    (paths relative to data/raw/landslide/).
    """
    labels_csv = os.path.join(config.RAW_LANDSLIDE_DIR, "labels.csv")
    if not os.path.exists(labels_csv):
        return [], [("landslide", "data/raw/landslide/labels.csv not found -- no real landslide "
                                   "source was reachable from this environment; see DATA_REPORT.md")]
    df = pd.read_csv(labels_csv)
    rows, skipped = [], []
    for _, r in df.iterrows():
        pre = os.path.join(config.RAW_LANDSLIDE_DIR, r["pre_image_path"])
        post = os.path.join(config.RAW_LANDSLIDE_DIR, r["post_image_path"])
        if not (os.path.exists(pre) and os.path.exists(post)):
            skipped.append((r["sample_id"], "listed in labels.csv but image file(s) missing"))
            continue
        severity = r["severity"]
        if severity not in config.SEVERITY_LEVELS:
            skipped.append((r["sample_id"], f"invalid severity '{severity}'"))
            continue
        rows.append({
            "sample_id": r["sample_id"],
            "disaster_event": r.get("source", "landslide"),
            "disaster_type": "Landslide",
            "severity": severity,
            "joint_label": f"Landslide_{severity}",
            "pre_image_path": os.path.relpath(pre, config.ROOT_DIR),
            "post_image_path": os.path.relpath(post, config.ROOT_DIR),
            "num_buildings": None,
            "num_classified_buildings": None,
            "sensor": r.get("sensor"),
            "gsd": r.get("gsd"),
            "capture_date": r.get("capture_date"),
            "source": r.get("source", "real supplementary landslide dataset"),
        })
    return rows, skipped


def coverage_report(df):
    counts = {c: 0 for c in config.JOINT_CLASSES}
    counts.update(df["joint_label"].value_counts().to_dict())
    report = {
        "target_per_class": config.MIN_SAMPLES_TARGET,
        "floor_per_class": config.MIN_SAMPLES_FLOOR,
        "classes": {},
    }
    for c in config.JOINT_CLASSES:
        n = counts.get(c, 0)
        report["classes"][c] = {
            "count": n,
            "meets_floor": n >= config.MIN_SAMPLES_FLOOR,
            "meets_target": n >= config.MIN_SAMPLES_TARGET,
        }
    report["all_meet_floor"] = all(v["meets_floor"] for v in report["classes"].values())
    report["all_meet_target"] = all(v["meets_target"] for v in report["classes"].values())
    return report


def print_coverage(report):
    print("\n=== Real-sample coverage per disaster+severity class "
          f"(floor={report['floor_per_class']}, target={report['target_per_class']}) ===")
    for c in config.JOINT_CLASSES:
        info = report["classes"][c]
        flag = "OK" if info["meets_floor"] else "SHORT"
        print(f"  {c:<20s} count={info['count']:<4d} [{flag}]")
    if not report["all_meet_floor"]:
        short = [c for c, v in report["classes"].items() if not v["meets_floor"]]
        print(f"\n{len(short)}/12 classes are BELOW the {report['floor_per_class']}-sample floor: {short}")
        print("See DATA_REPORT.md for exactly which sources were checked and why.")
    else:
        print("\nAll 12 classes meet the floor.")


def save_sample_grid(df, out_path, n=12):
    """Save a grid of real loaded pre/post images for visual confirmation
    that these are genuine satellite photos (buildings/terrain/roads), not
    placeholder/solid-color blocks."""
    sample = df.head(n)
    if len(sample) == 0:
        print("No samples available to render a sample grid.")
        return
    rows = len(sample)
    fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, 2)
    for i, (_, r) in enumerate(sample.iterrows()):
        for j, (col, tag) in enumerate([("pre_image_path", "PRE"), ("post_image_path", "POST")]):
            img = cv2.imread(os.path.join(config.ROOT_DIR, r[col]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"{r['sample_id']} {tag}\n{r['joint_label']}", fontsize=8)
            axes[i, j].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved real-image sample grid to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-grid-n", type=int, default=12)
    args = parser.parse_args()

    xbd_rows, xbd_skipped = scan_xbd()
    ls_rows, ls_skipped = scan_landslide()
    rows = xbd_rows + ls_rows
    skipped = xbd_skipped + ls_skipped

    if not rows:
        raise SystemExit(
            "No usable real samples found in data/raw/. Nothing to build metadata.csv from."
        )

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(config.METADATA_CSV), exist_ok=True)
    df.to_csv(config.METADATA_CSV, index=False)
    print(f"Wrote {len(df)} real samples to {config.METADATA_CSV}")

    if skipped:
        print(f"\n{len(skipped)} candidate label file(s)/rows were skipped:")
        for sid, reason in skipped:
            print(f"  - {sid}: {reason}")

    save_sample_grid(df, os.path.join(config.SAMPLES_DIR, "sample_grid.png"), n=args.sample_grid_n)

    report = coverage_report(df)
    print_coverage(report)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    with open(os.path.join(config.METRICS_DIR, "data_coverage_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
