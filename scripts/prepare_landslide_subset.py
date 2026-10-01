"""Pick a landslide-coverage-stratified subset of the real Landslide4Sense training data.

xBD has no landslide events, so the Landslide classes come from Landslide4Sense
(Ghorbanzadeh et al., 2022): 128x128 Sentinel-2 + ALOS PALSAR patches with binary
landslide masks. Only its training split has masks.

Run locally after downloading the Landslide4Sense training data:

    python prepare_landslide_subset.py path/to/TrainData.zip path/to/xbd-data

The input can be the downloaded .zip or the already-extracted folder. Patches are
grouped by the fraction of landslide pixels in their mask, and up to --per-bin
patches are copied from each group so every severity level has real samples.
"""
import argparse
import io
import os
import random
import re
import sys
import zipfile
from collections import defaultdict

import h5py

# Upper edges of landslide-pixel-fraction groups; patches with no landslide pixels are skipped.
BIN_EDGES = [0.02, 0.05, 0.10, 0.20, 0.30, 1.01]


def list_sources(path):
    """Yield (patch_id, kind, opener) for every image_*/mask_* .h5 in a zip or folder."""
    pattern = re.compile(r"(image|mask)_(\d+)\.h5$")
    if zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path)
        for name in zf.namelist():
            m = pattern.search(name)
            if m:
                yield int(m.group(2)), m.group(1), (lambda n=name: io.BytesIO(zf.read(n)))
    else:
        for root, _, files in os.walk(path):
            for name in files:
                m = pattern.search(name)
                if m:
                    full = os.path.join(root, name)
                    yield int(m.group(2)), m.group(1), (lambda f=full: open(f, "rb"))


def read_array(opener):
    with opener() as fh, h5py.File(fh, "r") as h5:
        key = list(h5.keys())[0]
        return h5[key][()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="Landslide4Sense training .zip or extracted folder")
    p.add_argument("out_dir", help="data repo folder; files go to <out_dir>/landslide4sense/{img,mask}")
    p.add_argument("--per-bin", type=int, default=50, help="max patches per landslide-fraction group")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    patches = defaultdict(dict)
    for pid, kind, opener in list_sources(args.source):
        patches[pid][kind] = opener
    pairs = {pid: v for pid, v in patches.items() if "image" in v and "mask" in v}
    if not pairs:
        sys.exit("No image_*.h5 / mask_*.h5 pairs found - is this the Landslide4Sense TRAINING data?")
    print(f"Found {len(pairs)} image/mask pairs. Measuring landslide coverage...")

    bins = defaultdict(list)
    for i, (pid, v) in enumerate(sorted(pairs.items()), 1):
        frac = float((read_array(v["mask"]) > 0).mean())
        if frac > 0:
            bins[next(j for j, e in enumerate(BIN_EDGES) if frac <= e)].append((pid, frac))
        print(f"\r  {i}/{len(pairs)}", end="", flush=True)
    print()

    rng = random.Random(args.seed)
    img_dir = os.path.join(args.out_dir, "landslide4sense", "img")
    mask_dir = os.path.join(args.out_dir, "landslide4sense", "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    print("\nLandslide fraction      available  copied")
    lo = 0.0
    for j, hi in enumerate(BIN_EDGES):
        chosen = rng.sample(bins[j], min(args.per_bin, len(bins[j])))
        for pid, _ in chosen:
            for kind, folder in (("image", img_dir), ("mask", mask_dir)):
                with pairs[pid][kind]() as src, open(os.path.join(folder, f"{kind}_{pid}.h5"), "wb") as dst:
                    dst.write(src.read())
        print(f"  ({lo:.2f}, {min(hi, 1.0):.2f}]        {len(bins[j]):6d}  {len(chosen):6d}")
        lo = hi
    print(f"\nDone. Output in: {os.path.abspath(os.path.join(args.out_dir, 'landslide4sense'))}")


if __name__ == "__main__":
    main()
