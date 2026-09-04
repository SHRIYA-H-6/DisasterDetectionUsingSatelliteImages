# Disaster Detection Using Satellite Images

Multi-task deep learning system that classifies **disaster type**
(Flood / Earthquake / Wildfire / Landslide) and **damage severity**
(Low / Moderate / Severe) from real pre/post-disaster satellite image
pairs, built on the real **xBD (xView2)** dataset (Gupta et al., 2019).

**Read `DATA_REPORT.md` first.** The real dataset currently bundled here is
small (a handful of real, verified xBD samples) because this build
environment's network policy blocks Kaggle/HuggingFace/Zenodo/xview2.org —
every script below is fully working and correct, but the model is trained
and evaluated on a coverage-limited real subset until the full dataset is
downloaded (see `download_data.py`). No synthetic or placeholder image was
used anywhere — every pixel here is a real satellite photo; provenance for
each file is in `data/raw/SOURCES.md`.

## Architecture (Phase 1)

A single-stage, multi-task **pseudo-Siamese ResNet-18**:

- Two independently-weighted ResNet-18 encoders (pretrained on ImageNet,
  fine-tuned) — one encodes the pre-disaster image, one the post-disaster
  image.
- Their 512-d embeddings are combined as `[pre, post, post - pre]`
  (concatenation + difference vector) into a shared 512-d representation.
- Two classification heads read that shared representation: a 4-way
  disaster-type head and a 3-way severity head.
- Trained with `CE(disaster) + CE(severity)` (equal weighting) via Adam.

See `model.py`.

## Pipeline

```
preprocessing.py   # parse real xBD JSON labels -> data/metadata.csv, sample grid, coverage report
dataset.py          # PyTorch Dataset + stratified train/val/test split
model.py             # pseudo-Siamese ResNet-18, dual heads
train.py             # enforces the per-class floor, trains, saves models/disaster_model.pt + loss curve
evaluate.py          # 3 classification reports (disaster/severity/combined) + 3 confusion matrices
predict.py           # single pre/post pair inference
batch_report.py      # inference + report over every row in metadata.csv
demo.py               # visual demo on a few real samples -> outputs/samples/demo_predictions.png
download_data.py     # fetch the full real xBD + real landslide data once network access allows it
```

Run in order:

```bash
pip install -r requirements.txt
python preprocessing.py
python train.py            # add --strict to require the 10-sample floor per class
python evaluate.py
python demo.py
```

## Evaluation outputs (`outputs/metrics/`)

- `disaster_type_report.json` — 4-class report (Flood/Earthquake/Wildfire/Landslide)
- `severity_report.json` — 3-class report (Low/Moderate/Severe)
- `combined_report.json` — 12-class joint `{disaster}_{severity}` report
  (e.g. `Flood_Severe`), including `accuracy`, `macro avg`, `weighted avg`

All precision/recall/f1-score values are rounded to 10 decimal places, per
the project spec. Confusion matrices (4x4, 3x3, 12x12) are saved as heatmaps
in `outputs/figures/`, alongside the training/validation loss curve.

## Minimum-data-per-class rule

Every one of the 12 `{disaster}_{severity}` classes needs its own real
sample count (target 15-20, floor 10) — enforced in code
(`train.py::check_floor`, backed by `preprocessing.py::coverage_report`),
not just described. `train.py --strict` refuses to train if any class is
under the floor; the default mode warns and proceeds in a clearly-labeled
"coverage-limited" run so the pipeline stays demonstrable while the dataset
grows. Current status: see `DATA_REPORT.md`.

## Phase 2 (planned, not yet implemented)

- Expand `data/raw/` to the full xBD dataset + a real landslide source via
  `download_data.py`, until every one of the 12 classes clears the floor.
- Literature-informed architecture upgrades once data volume supports them:
  a shared-weight (true Siamese) variant and an EfficientNet/ResNet-50
  backbone ablation, attention-based fusion of the pre/post embeddings
  (instead of concat+difference) as in later xView2 challenge winners, and
  class-weighted / focal loss to handle the natural class imbalance in xBD
  (`no-damage` vastly outnumbers `destroyed`).
- Building-level (not just tile-level) severity via the localization masks
  already present in the xBD JSON, for a segmentation + classification
  ensemble closer to the original xView2 challenge formulation.
