# Data coverage report

**Status: 3/12 classes (all of Earthquake) meet the floor with real data;
9/12 (all of Flood, Wildfire, Landslide) do not.** This is not a final
result — it's a status report, produced instead of padding gaps with fake
data, per the project's own "stop and report" rule.

## Per-class real-sample counts (target 15-20, floor 10)

| Disaster + Severity | Real samples found | Meets floor (10)? |
|---|---|---|
| Flood_Low | 3 | No |
| Flood_Moderate | 0 | No |
| Flood_Severe | 2 | No |
| Earthquake_Low | 20 | **Yes** |
| Earthquake_Moderate | 20 | **Yes** |
| Earthquake_Severe | 20 | **Yes** |
| Wildfire_Low | 0 | No |
| Wildfire_Moderate | 0 | No |
| Wildfire_Severe | 0 | No |
| Landslide_Low | 0 | No |
| Landslide_Moderate | 0 | No |
| Landslide_Severe | 0 | No |

(Regenerate this table any time with `python preprocessing.py` — it prints
and saves the live count to `outputs/metrics/data_coverage_report.json`.)

One additional real image (`santa-rosa-wildfire_00000007`, Wildfire) was
found with a matching official xBD label but **no pre-disaster
counterpart**, so it isn't counted above; see `data/raw/SOURCES.md`.

## The earthquake breakthrough: Maxar Open Data

The links you supplied (ITU CSCRS's KATE-PD page, its IGARSS 2025 paper,
and its Google share link) are all on `web.cscrs.itu.edu.tr`,
`ieeexplore.ieee.org`, and `share.google` — all network-blocked in this
sandbox, same as before (see below). But following the trail from the
KATE-PD paper (which itself sources imagery from **Maxar Open Data** and
**Airbus Pleiades**) led somewhere reachable: **`maxar-opendata.s3.amazonaws.com`
is a public, unauthenticated S3 bucket, and `s3.amazonaws.com` is not
network-blocked here.**

That bucket hosts real, high-resolution pre/post satellite imagery for the
2023 Kahramanmaras, Turkiye earthquake (and dozens of other real disaster
events — see below), **plus Maxar's own official building-change damage
layer** (`events/Kahramanmaras-turkey-earthquake-23/building_change/*.gpkg`)
— AI-detected building-change polygons between real pre/post image pairs,
each typed "removed", "changed", "new", or "no change" by Maxar.

`fetch_maxar_earthquake.py` (new, and it actually **runs successfully in
this sandbox** — verified this session):
1. Downloads the two building-change GeoPackages (zones 36 & 37 covering
   the affected provinces).
2. Cross-references their quadkeys against which ones actually have both a
   real pre-event and real post-event image on S3 (79 did).
3. Computes `damage_ratio = removed/changed polygons ÷ all polygons` per
   quadkey — a real, Maxar-published per-tile damage signal — and bins
   quadkeys into Low/Moderate/Severe by tertiles.
4. Reads each pre/post image at 1024×1024 directly from the source
   Cloud-Optimized GeoTIFFs over HTTP range requests (no need to download
   the full ~20-100MB source file per tile).

Result: **60 real earthquake samples, exactly 20/20/20 across
Low/Moderate/Severe** — comfortably clearing the 15-20 target, not just the
floor, for all three Earthquake classes. Full provenance (exact quadkey,
acquisition dates, damage ratio) is in
`data/raw/maxar_earthquake/labels.csv`; methodology and honesty notes on
label confidence are in `data/raw/SOURCES.md`.

This label source is coarser than xBD's (a tile-level ratio derived from
Maxar's own AI building-change detector, vs. xBD's manual per-building
annotation) — worth knowing if you're citing per-sample confidence, but it
is a real, published, per-tile signal, not an invented one.

## Wildfire and landslide: real imagery found, real severity still blocked

The same Maxar Open Data bucket also has real pre/post imagery for
several real wildfire and landslide events:

- Wildfire: `WildFires-LosAngeles-Jan-2025`, `Maui-Hawaii-fires-Aug-23`,
  `SmokeHouseCreek-Wildfires-Texas-Mar24`, `Belize-Wildfires-June24`,
  `McDougallCreekWildfire-BC-Canada-Aug-23`, and others. 40 quadkeys in the
  LA event alone were confirmed to have real pre+post pairs; post-event
  smoke is visibly present in some tiles (visually verified this session).
- Landslide: `PNG-Landslide-June24` (Papua New Guinea) and
  `shovi-georgia-landslide-8Aug23` (Georgia).

**But neither disaster type has an official per-tile damage/burn-severity
layer** the way the earthquake event has `building_change` — that product
was specific to Maxar's Turkiye earthquake response effort. Attempts to
derive one this session:

- **Wildfire burn severity**: a real, standard remote-sensing technique
  (dNDVI, a simplified cousin of the operational USGS/USFS dNBR method)
  was tried using Maxar's 8-band multispectral (`-ms.tif`) pre/post pairs
  (Red + NIR bands). On a first test tile it produced a near-zero,
  not-clearly-meaningful signal — likely because whole-tile-average NDVI on
  raw, non-atmospherically-corrected digital numbers is too noisy/diluted
  by unburned area within the tile. This needs proper calibration (or a
  tighter crop to the actual burned footprint) to trust as a real label, so
  it was **not** used to populate `metadata.csv` — better to report the gap
  than ship an unvalidated number as a "real" severity label. The relevant
  real, per-structure severity data does exist publicly (CAL FIRE's DINS
  damage-inspection database, `data.ca.gov`/`gis.data.cnra.ca.gov`), but
  those hosts are network-blocked here (see below).
- **Landslide severity**: no equivalent official layer or public damage
  database was found reachable at all.

Phase 2 next step: either invest in a properly calibrated dNBR pipeline (or
manual/expert labeling) against the real Maxar wildfire/landslide imagery
already identified above, or unblock `data.ca.gov` (CAL FIRE DINS) /
Zenodo / HuggingFace and pull an existing labeled dataset directly (see
below).

## Why xview2.org / Kaggle / HuggingFace / Zenodo / share.google / ieeexplore.ieee.org are still blocked

This session runs in a sandboxed Claude Code environment whose network
policy ("Default – trusted network access") allows `github.com` (git
clone), `pypi.org`/`npmjs.org` (package installs), and — critically —
generic `s3.amazonaws.com` / `storage.googleapis.com` endpoints, but
**blocks (`403`/`connect_rejected` at the TLS CONNECT stage) every one of
the following**, verified directly from this session (including the three
links you supplied):

| Host | Purpose | Result |
|---|---|---|
| `xview2.org` | Official xBD dataset (canonical source) | blocked |
| `www.kaggle.com` | Kaggle mirrors of xView2/xBD | blocked |
| `huggingface.co`, `cdn-lfs*.huggingface.co` | HuggingFace dataset mirrors | blocked |
| `zenodo.org` | Zenodo-hosted landslide inventories (Bijie, CAS Landslide, RER2023, etc.) | blocked |
| `drive.google.com` | Google Drive mirrors some projects use | blocked |
| `web.cscrs.itu.edu.tr` | ITU CSCRS's KATE-PD dataset page (your link) | blocked |
| `ieeexplore.ieee.org` | The KATE-PD IGARSS 2025 paper (your link) | blocked |
| `share.google` | Your third link | blocked |
| `codeocean.com`, `git.codeocean.com` | KATE-PD's primary data host | blocked |
| `data.ca.gov`, `gis.data.cnra.ca.gov`, `*.arcgis.com` | CAL FIRE DINS wildfire damage database | blocked |
| `pan.baidu.com` | Hosts at least one real landslide dataset (LRSTTC) | blocked |

`raw.githubusercontent.com`, `codeload.github.com`, `s3.amazonaws.com`, and
`storage.googleapis.com` all respond normally (not connection-rejected),
and `git clone https://github.com/...` works fully. **This is the entire
reason the earthquake breakthrough above was possible**: KATE-PD's own
GitHub repo (`github.com/CSCRS/kate-pd`) doesn't carry the imagery (too
large for git, hosted on CodeOcean/HuggingFace/Zenodo instead — all
blocked), but its README pointed at Maxar Open Data as the imagery source,
and that happens to be a plain public S3 bucket.

**To fix this properly:** change the session's environment network policy
to allow-list the hosts above (via claude.ai/code → Settings →
Environments, or a new environment with a custom network policy — see
https://code.claude.com/docs/en/claude-code-on-the-web), then run
`download_data.py` in a fresh session on that environment for the full xBD
dataset (clears Flood/Wildfire's remaining gap directly) plus a real
landslide/wildfire-severity source.

## What GitHub scavenging found for xBD pixels directly (unchanged from before)

15+ public GitHub repositories were cloned and inspected for genuine,
committed xBD pixel data. Findings (see `data/raw/SOURCES.md` for full
detail): `jeremiahf24/ADS-Capstone` and `SmashNdashH/CSE499A.15-Msrb-` have
the complete official xBD label JSON for many disasters but almost no
image tiles (expected — xBD imagery is ~1GB+, typically `.gitignore`d).
6 real hurricane-florence pre/post pairs with matching labels were
recovered from `jeremiahf24/ADS-Capstone` (Flood). No genuine, bulk,
committed earthquake or wildfire xBD image tiles were found on GitHub, and
no genuine landslide imagery via GitHub at all — the earthquake gap above
was closed via Maxar Open Data instead, not xBD.

## What's needed to actually complete this dataset

1. **Unblock network access** (see table above), then run:
   ```
   python download_data.py --method official --url "<your xview2.org signed URL>"
   python download_data.py --method kaggle --kaggle-dataset "<owner>/<slug>"
   python download_data.py --method huggingface --hf-repo "<namespace>/<name>"
   ```
   The full xBD dataset directly covers Flood and Wildfire with hundreds of
   tiles per event — pulling it should clear those 6 remaining classes.
2. **Landslide and Wildfire severity**: either (a) unblock
   `data.ca.gov`/`gis.data.cnra.ca.gov` and pair CAL FIRE's real DINS
   per-structure damage database with the Maxar LA-wildfire imagery already
   identified, (b) unblock Zenodo/HuggingFace and pull CAS Landslide /
   Bijie / Landslide4Sense, or (c) invest in a properly calibrated dNBR
   pipeline against the real Maxar multispectral imagery already
   identified in this session (`fetch_maxar_earthquake.py` is a working
   template for the S3 + COG + GeoPackage plumbing).
3. Re-run `python preprocessing.py` — it re-scans `data/raw/` automatically
   (xBD + any `data/raw/<name>/labels.csv` extra source, including
   `data/raw/maxar_earthquake/`), regenerates `data/metadata.csv`, and
   reprints the coverage table with no code changes needed.
4. Once `all_meet_floor` is true in
   `outputs/metrics/data_coverage_report.json`, `python train.py --strict`
   will proceed instead of warning, and `python evaluate.py`'s three
   reports will have real, non-zero support in every one of the 12 classes.

## What *is* fully working right now (Phase 1 code, proven on real data)

Every script (`preprocessing.py`, `dataset.py`, `model.py`, `train.py`,
`evaluate.py`, `predict.py`, `batch_report.py`, `demo.py`,
`download_data.py`, `fetch_maxar_earthquake.py`) runs end-to-end today
against 65 real samples (5 real xBD hurricane-florence flood samples + 60
real Maxar earthquake samples, all 3 severities): real images load and are
visually confirmed non-synthetic (`outputs/samples/sample_grid.png`), the
pseudo-Siamese ResNet-18 trains and checkpoints, and `evaluate.py` produces
all three required report files with all 12/4/3 classes present (10-decimal
precision/recall/f1). The per-class floor check runs automatically before
training and would abort training under `--strict` for the 9 still-short
classes — it isn't just described here, it's enforced in code
(`train.py::check_floor`). Once real coverage is high enough for the
remaining 9 classes, the entire pipeline runs unchanged at full scale; only
`data/raw/` needs to grow.
