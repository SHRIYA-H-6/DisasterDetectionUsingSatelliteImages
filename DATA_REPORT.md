# Data coverage report

**Status: below the required per-class floor. This is not a final result —
it's a status report, produced instead of padding gaps with fake data, per
the project's own "stop and report" rule.**

## Per-class real-sample counts (target 15-20, floor 10)

| Disaster + Severity | Real samples found | Meets floor (10)? |
|---|---|---|
| Flood_Low | 3 | No |
| Flood_Moderate | 0 | No |
| Flood_Severe | 2 | No |
| Earthquake_Low | 0 | No |
| Earthquake_Moderate | 0 | No |
| Earthquake_Severe | 0 | No |
| Wildfire_Low | 0 | No |
| Wildfire_Moderate | 0 | No |
| Wildfire_Severe | 0 | No |
| Landslide_Low | 0 | No |
| Landslide_Moderate | 0 | No |
| Landslide_Severe | 0 | No |

**12 / 12 classes are below the floor.** (Regenerate this table any time
with `python preprocessing.py` — it prints and saves the live count to
`outputs/metrics/data_coverage_report.json`.)

One additional real image (`santa-rosa-wildfire_00000007`, Wildfire/Severe)
was found with a matching official xBD label but **no pre-disaster
counterpart**, so it cannot form a pre/post pair and isn't counted above; see
`data/raw/SOURCES.md`.

## Why: every standard xBD distribution channel is network-blocked here

This session runs in a sandboxed Claude Code environment whose network
policy ("Default – trusted network access") allows `github.com` (git
clone), `pypi.org`/`npmjs.org` (package installs), and generic
`s3.amazonaws.com` / `storage.googleapis.com` endpoints — and explicitly
**blocks (`403`/`connect_rejected` at the TLS CONNECT stage) every one of
the following**, verified directly from this session:

| Host | Purpose | Result |
|---|---|---|
| `xview2.org` | Official xBD dataset (canonical source) | blocked |
| `www.kaggle.com` | Kaggle mirrors of xView2/xBD | blocked |
| `huggingface.co`, `cdn-lfs*.huggingface.co` | HuggingFace dataset mirrors | blocked |
| `zenodo.org` | Zenodo-hosted landslide inventories (Bijie, CAS Landslide, etc.) | blocked |
| `drive.google.com` | Google Drive mirrors some projects use | blocked |

`raw.githubusercontent.com`, `codeload.github.com`, `s3.amazonaws.com`, and
`storage.googleapis.com` all responded (with ordinary HTTP status codes, not
connection rejections), and `git clone https://github.com/...` works fully
— so real data recovery in this session was possible only through public
GitHub repositories.

**To fix this properly:** change the session's environment network policy to
allow-list `xview2.org`, `kaggle.com`/`www.kaggle.com`,
`huggingface.co`+`*.huggingface.co`, and `zenodo.org` (via
claude.ai/code → Settings → Environments, or a new environment with a
custom network policy — see
https://code.claude.com/docs/en/claude-code-on-the-web), then run
`download_data.py` (see below) in a fresh session on that environment. Until
then, expansion is limited to what's reachable via `git clone` of public
GitHub repos.

## What GitHub scavenging actually found (and why most of it doesn't count)

15+ public GitHub repositories were cloned and inspected for genuine,
committed xBD pixel data (not just code that *references* xBD). Findings:

- **`jeremiahf24/ADS-Capstone`** and **`SmashNdashH/CSE499A.15-Msrb-`** —
  course-project repos containing the **complete official xBD label JSON**
  for many disasters (hurricane-harvey/florence/matthew/michael,
  midwest-flooding, palu-tsunami, mexico-earthquake, santa-rosa-wildfire,
  socal-fire, guatemala-volcano) — thousands of files — but almost none of
  the corresponding image tiles. This is expected: xBD imagery is ~1GB+, so
  student repos typically `.gitignore` the pixels and commit only the
  lightweight label JSON.
- **`jeremiahf24/ADS-Capstone`** did commit 6 real hurricane-florence
  pre/post PNG pairs alongside their matching labels → used here (Flood).
- **`Chimdiya1/Webapp`** committed 3 real hurricane-harvey pre/post pairs
  (as 512x512 sub-crops for a web demo) but no matching label could be
  verified against the crop geometry → excluded (see `SOURCES.md`).
- **`DIUx-xView/xview2-baseline`** (the official xView2 baseline repo)
  ships one real post-disaster santa-rosa-wildfire demo image; its
  pre-disaster counterpart isn't in that repo or any other reachable
  source.
- **`ashnair1/xview2-toolkit`** has real palu-tsunami files, but they are
  colorized segmentation-mask renders (verified via PIL image mode
  `P`/`1`, not RGB), not raw satellite photos — excluded per the "must look
  like a real satellite photo, not a flat-color block" rule.
- No repo with genuine, bulk, committed **earthquake** or **wildfire**
  image tiles (beyond the single wildfire fragment above) was found.
- No repo with genuine **landslide** satellite/aerial imagery (Bijie, CAS
  Landslide, Landslide4Sense, or otherwise) was found committed to GitHub;
  every project referencing these datasets links out to Kaggle/Zenodo/HF,
  which are blocked here.

Full per-file provenance (exact repo, path, and why each file was
included/excluded) is in `data/raw/SOURCES.md`.

## What's needed to actually complete this dataset

1. **Unblock network access** (see above), then run one of:
   ```
   python download_data.py --method official --url "<your xview2.org signed URL>"
   python download_data.py --method kaggle --kaggle-dataset "<owner>/<slug>"
   python download_data.py --method huggingface --hf-repo "<namespace>/<name>"
   python download_data.py --method landslide --landslide-source bijie --url "<confirmed URL>"
   ```
   xBD alone directly covers Flood, Earthquake (+tsunami), and Wildfire —
   pulling the full dataset should clear those 9 classes past the floor
   immediately, since xBD has hundreds of tiles per disaster event.
2. **Landslide still needs a separate real source** (xBD has none) — a real
   published inventory with imagery (Bijie / CAS Landslide / Landslide4Sense
   are the standard choices; see `download_data.py`'s docstring), with
   severity derived from that source's own damage/susceptibility
   annotations and tagged with its own `source` value in `metadata.csv`
   (already supported by `preprocessing.py`'s `scan_landslide()`).
3. Re-run `python preprocessing.py` — it re-scans `data/raw/` automatically,
   regenerates `data/metadata.csv`, and reprints the coverage table with no
   code changes needed.
4. Once `all_meet_floor` is true in
   `outputs/metrics/data_coverage_report.json`, `python train.py --strict`
   will proceed instead of warning, and `python evaluate.py`'s three
   reports will have real, non-zero support in every one of the 12 classes.

## What *is* fully working right now (Phase 1 code, proven on real data)

Every script (`preprocessing.py`, `dataset.py`, `model.py`, `train.py`,
`evaluate.py`, `predict.py`, `batch_report.py`, `demo.py`,
`download_data.py`) runs end-to-end today against the 5 real, verified
hurricane-florence samples: real images load and are visually confirmed
non-synthetic (`outputs/samples/sample_grid.png`), the pseudo-Siamese
ResNet-18 trains and checkpoints, and `evaluate.py` produces all three
required report files with all 12/4/3 classes present (10-decimal
precision/recall/f1, zero-filled where support is currently 0) and all
three confusion matrices. The per-class floor check runs automatically
before training and would abort training under `--strict` — it isn't just
described here, it's enforced in code (`train.py::check_floor`). Once
real coverage is high enough, the entire pipeline runs unchanged at full
scale; only `data/raw/` needs to grow.
