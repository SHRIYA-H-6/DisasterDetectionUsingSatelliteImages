# Disaster Detection from Satellite Images

Neural Networks and Deep Learning (23CSE473) course project, Team A1 (CSE A).

Given a pair of pre- and post-event satellite images, the system:

1. classifies the **disaster type** (Flood, Earthquake, Tsunami, Landslide) and **severity** (Low, Moderate, Severe), giving 12 combined classes;
2. segments the **affected regions** with a U-Net and measures the **affected area %** from that mask;
3. draws the affected regions on a **Matplotlib overlay** and an interactive **Folium map** at real coordinates;
4. writes an automated **summary report** with a recommended emergency action.

All images are real. No synthetic or placeholder data is used anywhere.

## Data

| Disaster type | Source | Events | Severity measure |
|---|---|---|---|
| Flood | xBD (Gupta et al., 2019), challenge training set | `midwest-flooding` | share of damaged buildings in the tile |
| Earthquake | xBD + BRIGHT (Chen et al., 2025) | `mexico-earthquake` + BRIGHT `*earthquake*` events | share of damaged buildings / building pixels |
| Tsunami | xBD | `palu-tsunami` | share of damaged buildings in the tile |
| Landslide | Landslide4Sense (Ghorbanzadeh et al., 2022), training split | Sentinel-2 patches | share of landslide pixels in the patch |

Severity thresholds: buildings Low ≤ 10% < Moderate ≤ 30% < Severe. Landslide pixels Low ≤ 5% < Moderate ≤ 20% < Severe.

The datasets cannot be redistributed, so they live in a separate **private** repo (`xbd-data`), cloned next to this one. These scripts extract the needed subsets from the official downloads:

```
python scripts/prepare_xbd_subset.py      <xview2 train tar.gz>      <xbd-data folder>
python scripts/prepare_bright_subset.py   <xbd-data folder>          <BRIGHT zip(s) or folder>
python scripts/prepare_landslide_subset.py <Landslide4Sense train zip> <xbd-data folder>
```

## Models

* **Classifier:** pseudo-Siamese multitask EfficientNet-B0. Two separate ImageNet-pretrained encoders (pre / post). Their pooled features and the difference between them feed a dropout-regularised shared layer, then two softmax heads (type, severity). The loss is the sum of the two cross-entropies, with equal weights. Optimiser: Adam.
* **Segmentation:** U-Net on the stacked pre/post pair. It predicts background / intact structure / affected (damaged building or landslide) per pixel, and is trained on the datasets' pixel-level labels.

Overfitting safeguards (classifier; most also apply to the U-Net):

* dropout in the heads;
* Adam weight decay (L2);
* flip / rotation / brightness-contrast augmentation on the training split only;
* class-balanced sampling;
* early stopping on validation loss (patience 7) with the best weights restored;
* progressive unfreezing of the encoders;
* an explicit overfitting detector that stops training when validation loss rises while training loss falls;
* stratified k-fold cross-validation;
* the train/val loss gap reported in the final evaluation.

## Running

```
pip install -r requirements.txt
python -m disaster_detection.build_metadata        # stage 1: labels, splits, minimum-sample check
python -m disaster_detection.preprocess            #          cache 256px tiles
python -m disaster_detection.train_classifier --cv-folds 5
python -m disaster_detection.train_segmentation
python -m disaster_detection.evaluate              # 3 reports (console + JSON), confusion matrices, CSV
python -m disaster_detection.demo                  # end-to-end on real test images -> report + overlay + map
python -m disaster_detection.predict --pre a.png --post b.png   # any new image pair
```

The data location defaults to `../xbd-data`. Override it with `DISASTER_DATA_ROOT`. Outputs go to `outputs/`.

## Known limitations

* Earthquake Moderate/Severe samples come from BRIGHT, whose post-event images are SAR, not optical. The model can partly tell those samples apart by modality.
* Landslide4Sense patches are 10 m Sentinel-2 imagery with no pre-event image (pre = post). xBD is about 0.5 m optical. Imaging differences can therefore help the type head.
* xBD labels damaged buildings, not floodwater, so "affected area" for xBD tiles means the area of damaged structures.
