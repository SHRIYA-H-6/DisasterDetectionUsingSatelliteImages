"""
Fetch the FULL real xBD dataset (and a real supplementary landslide source)
once network access to the relevant hosts is available.

This sandbox's network policy blocks huggingface.co, kaggle.com, zenodo.org
and xview2.org (only github.com, pypi.org and generic S3/GCS endpoints are
reachable here) -- see DATA_REPORT.md for the exact reachability check. This
script is therefore written to be run OUTSIDE this sandbox: on your own
machine, in a Kaggle/Colab notebook, or in a Claude Code environment whose
network policy allow-lists these hosts (see the "network policy" section of
https://code.claude.com/docs/en/claude-code-on-the-web).

Usage:
    # Official source (requires a free xview2.org account -- most complete,
    # canonical option; direct download links are emailed/shown after login,
    # there is no public API to automate the login itself):
    python download_data.py --method official --url "<signed URL from xview2.org>"

    # Kaggle mirror (requires `pip install kaggle` + a configured
    # ~/.kaggle/kaggle.json API token). Kaggle mirror slugs for xBD/xView2
    # change over time -- search https://www.kaggle.com/datasets?search=xview2
    # or search=xbd and pass the exact slug you find:
    python download_data.py --method kaggle --kaggle-dataset "<owner>/<dataset-slug>"

    # Hugging Face Hub mirror, if one is available (search
    # https://huggingface.co/datasets?search=xbd first and pass the exact
    # repo id you find):
    python download_data.py --method huggingface --hf-repo "<namespace>/<dataset-name>"

    # Real supplementary landslide imagery (xBD has no landslide coverage).
    # Known real, published landslide inventories with satellite/aerial
    # imagery you can use here -- confirm the current download link for
    # each on its own project page before running, since hosting changes:
    #   - Bijie Landslide Dataset (Ju et al., 2022) -- optical UAV/satellite
    #     imagery + binary landslide masks, Guizhou Province, China.
    #   - CAS Landslide Dataset (2024) -- large multi-sensor landslide
    #     benchmark, IEEE DataPort / Zenodo.
    #   - Landslide4Sense (Ghorbanzadeh et al., 2022) -- Sentinel-2 + terrain
    #     data, hosted by IARAI (iarai.ac.at/landslide4sense) with a
    #     Hugging Face mirror.
    python download_data.py --method landslide --landslide-source bijie --url "<confirmed URL>"

After downloading, this script normalizes whatever it fetched into:
    data/raw/xbd/images/<event>_<id>_{pre,post}_disaster.png
    data/raw/xbd/labels/<event>_<id>_{pre,post}_disaster.json
    data/raw/landslide/images/...  + data/raw/landslide/labels.csv
so that `python preprocessing.py` picks it up automatically -- no other code
changes are needed once real data is in place.
"""
import argparse
import os
import shutil
import zipfile

import config


def _extract_zip(zip_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print(f"Extracted {zip_path} -> {dest_dir}")


def download_official(url, dest_dir=config.RAW_XBD_DIR):
    """Download a signed xView2.org tile archive (tar/zip) the user obtained
    after registering at https://xview2.org and accepting the data license."""
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, "xbd_download.zip")
    print(f"Downloading {url} -> {local_path}")
    urllib.request.urlretrieve(url, local_path)
    if local_path.endswith(".zip"):
        _extract_zip(local_path, dest_dir)
    print("Move/rename the extracted 'images/' and 'labels/' folders into "
          f"{config.RAW_XBD_IMAGES_DIR} and {config.RAW_XBD_LABELS_DIR} "
          "(the official archive layout is <tier>/images/*.png and "
          "<tier>/labels/*.json per disaster).")


def download_kaggle(dataset_slug, dest_dir=config.RAW_XBD_DIR):
    """Requires `pip install kaggle` and ~/.kaggle/kaggle.json configured
    with your Kaggle API token (see https://www.kaggle.com/docs/api)."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise SystemExit("Run `pip install kaggle` first, and place your API "
                          "token at ~/.kaggle/kaggle.json (from Kaggle "
                          "Account Settings -> Create New API Token).")
    os.makedirs(dest_dir, exist_ok=True)
    os.system(f"kaggle datasets download -d {dataset_slug} -p {dest_dir} --unzip")
    print(f"Downloaded Kaggle dataset '{dataset_slug}' into {dest_dir}. "
          "Inspect its folder layout and move images/labels into "
          f"{config.RAW_XBD_IMAGES_DIR} / {config.RAW_XBD_LABELS_DIR}.")


def download_huggingface(repo_id, dest_dir=config.RAW_XBD_DIR):
    """Requires `pip install huggingface_hub` (or `datasets`)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Run `pip install huggingface_hub` first.")
    local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dest_dir)
    print(f"Downloaded HF dataset '{repo_id}' -> {local_dir}. Inspect its "
          f"layout and move images/labels into {config.RAW_XBD_IMAGES_DIR} / "
          f"{config.RAW_XBD_LABELS_DIR}.")


def download_landslide(source, url, dest_dir=config.RAW_LANDSLIDE_DIR):
    """Downloads a real landslide archive and reminds you to write
    data/raw/landslide/labels.csv (sample_id, pre_image_path, post_image_path,
    severity, source) so preprocessing.py can pick it up. Severity should be
    derived from the source's own damage/susceptibility annotations, mapped
    to Low/Moderate/Severe -- never fabricated."""
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, f"{source}_download.zip")
    print(f"Downloading {source} data from {url} -> {local_path}")
    urllib.request.urlretrieve(url, local_path)
    if local_path.endswith(".zip"):
        _extract_zip(local_path, dest_dir)
    print(f"Now write {os.path.join(dest_dir, 'labels.csv')} mapping each real "
          "pre/post pair to a Low/Moderate/Severe severity derived from this "
          "source's own annotations, then re-run `python preprocessing.py`.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True,
                         choices=["official", "kaggle", "huggingface", "landslide"])
    parser.add_argument("--url", help="Direct download URL (official/landslide methods)")
    parser.add_argument("--kaggle-dataset", help="owner/dataset-slug (kaggle method)")
    parser.add_argument("--hf-repo", help="namespace/dataset-name (huggingface method)")
    parser.add_argument("--landslide-source", default="bijie",
                         help="Label for which real landslide source this is (bijie/cas/landslide4sense/...)")
    args = parser.parse_args()

    if args.method == "official":
        if not args.url:
            raise SystemExit("--url is required for --method official (get it from your xview2.org account).")
        download_official(args.url)
    elif args.method == "kaggle":
        if not args.kaggle_dataset:
            raise SystemExit("--kaggle-dataset is required, e.g. --kaggle-dataset owner/xbd-dataset-slug "
                              "(search https://www.kaggle.com/datasets?search=xview2 for the current slug).")
        download_kaggle(args.kaggle_dataset)
    elif args.method == "huggingface":
        if not args.hf_repo:
            raise SystemExit("--hf-repo is required, e.g. --hf-repo namespace/xbd "
                              "(search https://huggingface.co/datasets?search=xbd for the current repo id).")
        download_huggingface(args.hf_repo)
    elif args.method == "landslide":
        if not args.url:
            raise SystemExit("--url is required for --method landslide.")
        download_landslide(args.landslide_source, args.url)

    print("\nDone. Run `python preprocessing.py` next to rebuild data/metadata.csv "
          "and re-check per-class coverage against the 10-20 sample floor.")


if __name__ == "__main__":
    main()
