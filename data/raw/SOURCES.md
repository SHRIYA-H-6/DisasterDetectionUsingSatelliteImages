# Data provenance

All imagery and labels in `data/raw/` are **real xBD (xView2) satellite tiles and
official xBD damage-annotation JSON** (Gupta et al., 2019, "Creating xBD: A
Dataset for Assessing Building Damage from Satellite Imagery"). No synthetic,
generated, or placeholder images are used anywhere in this repository.

## Why this set is small

This build environment's outbound network policy allows `github.com` (git
clone), `pypi.org`/`npmjs.org` (package installs), and generic `s3.amazonaws.com`
/ `storage.googleapis.com` endpoints, but **blocks `huggingface.co`,
`www.kaggle.com`, `zenodo.org`, and `xview2.org`** — the official/standard
distribution channels for the full xBD dataset (~11,000 tile pairs, ~1GB+) and
for real landslide inventories. See `DATA_REPORT.md` at the repo root for the
full per-class coverage accounting and what is needed to complete the dataset.

Given that restriction, the images below were recovered from public GitHub
repositories that had incidentally committed small, genuine fragments of xBD
(coursework/research repos). Each entry names the exact upstream repo the
files came from, cloned during this session (2026-09-04).

## Files and their origin

### `hurricane-florence_00000086/88/91/93/94/96` (pre + post, PNG 1024x1024, + matching official JSON labels)
- Disaster: xBD `disaster_type = "flooding"` → mapped to **Flood**
- Source repo: https://github.com/jeremiahf24/ADS-Capstone
  - Images: `Images/hurricane-florence_<id>_{pre,post}_disaster.png`
  - Labels: `Data/hurricane-florence_<id>_{pre,post}_disaster.json`
- Verified: image dims (1024x1024) match `original_width`/`original_height`
  in the JSON metadata, and JSON `metadata.sensor`, `gsd`, `capture_date`,
  `catalog_id` fields match the real xBD schema exactly (GEOEYE01 sensor,
  2018-09-20 capture date, consistent with the real Hurricane Florence xBD
  collection). Building-level `properties.subtype` values
  (no-damage/minor-damage/major-damage/destroyed) are used to derive
  tile-level severity in `preprocessing.py`.

### `santa-rosa-wildfire_00000007` (post only, PNG 1024x1024, + matching official JSON label)
- Disaster: xBD `disaster_type = "fire"` → mapped to **Wildfire**
- Image source: https://github.com/DIUx-xView/xview2-baseline
  (`overlay_output_to_image/santa-rosa-wildfire_00000007_post_disaster.png`
  — the official xView2 baseline repo's own demo image, itself a genuine xBD
  tile used to illustrate their scoring/overlay tool)
- Label source: https://github.com/SmashNdashH/CSE499A.15-Msrb-
  (`data/processed_labels/santa-rosa-wildfire_00000007_post_disaster.json`)
- **No pre-disaster counterpart image was found on any reachable source.**
  This sample is included only for the demo script / sample-grid visual
  check, and is excluded from training (a pseudo-Siamese model requires a
  pre/post pair). It does not count toward any class's sample floor.

## Explicitly NOT included (and why)

- `hurricane-harvey_00000000/1/2` patches from
  https://github.com/Chimdiya1/Webapp (`public/images/{pre,post}/...`) —
  real xBD pixels, but they are 512x512 sub-crops produced by that repo's
  own web-app pipeline, and no JSON label could be verified to align with
  the crop geometry. Rather than guess a severity label, these are left out
  of `metadata.csv` entirely (see `DATA_REPORT.md`).
- `palu-tsunami_00000001` files from
  https://github.com/ashnair1/xview2-toolkit (`.github/*_seg.png`) — these
  are colorized *segmentation-mask* renders (PIL mode `P`/`1`, palette /
  binary), not raw satellite RGB imagery, so they fail the "real satellite
  photo, not a flat-color block" visual check by construction. Excluded.
- Earthquake and landslide: no real, verifiable pixel data reachable from
  this sandbox at all (see `DATA_REPORT.md`).

## Full xBD official label set (reference only, not used for pixels)

`https://github.com/jeremiahf24/ADS-Capstone` and
`https://github.com/SmashNdashH/CSE499A.15-Msrb-` both contain (as course
project artifacts) the **complete official xBD JSON label files** for many
disasters (hurricane-harvey/florence/matthew/michael, midwest-flooding,
palu-tsunami, mexico-earthquake, santa-rosa-wildfire, socal-fire,
guatemala-volcano) — thousands of label files, but essentially none of the
corresponding image tiles (image binaries were `.gitignore`d in those
projects, as is typical since xBD imagery totals ~1GB+). These confirm the
label schema and disaster taxonomy used in `preprocessing.py`, but cannot by
themselves produce trainable image samples.
