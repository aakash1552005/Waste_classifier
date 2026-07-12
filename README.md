# Waste Classifier — Model Training

Training pipeline for the image classifier used by the
[AI-Based Smart Garbage Segregation Bin](#) project. Takes photos of waste
dropped into the bin's chute and classifies them into one of five
categories: **wet, dry, plastic, biomedical, recyclable**.

Trained in Google Colab on a T4 GPU using transfer learning on MobileNetV2.
Exports a `waste_classifier.h5` file that's consumed by the bin's Flask
inference server (see the main project repo).

---

## Table of contents

* [Results](#results)
* [Open in Colab](#open-in-colab)
* [Dataset](#dataset)
* [Class mapping](#class-mapping)
* [Preprocessing and augmentation](#preprocessing-and-augmentation)
* [Model architecture](#model-architecture)
* [Training procedure](#training-procedure)
* [Evaluation](#evaluation)
* [How to run this notebook](#how-to-run-this-notebook)
* [Output files](#output-files)
* [Using the model](#using-the-model)
* [Known limitations](#known-limitations)
* [Possible improvements](#possible-improvements)
* [License](#license)

---

## Results

Final model, evaluated on a held-out validation split (4,149 images):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Wet | 0.975 | 0.975 | 0.975 | 197 |
| Dry | 0.995 | 0.975 | 0.985 | 1,990 |
| Plastic | 0.850 | 0.919 | 0.883 | 173 |
| Biomedical | 0.987 | 1.000 | 0.994 | 1,232 |
| Recyclable | 0.938 | 0.955 | 0.947 | 557 |

**Overall validation accuracy: 97.76%** (macro avg F1: 0.957, weighted avg
F1: 0.978)

Plastic is the weakest class (fewest training examples — see
[Dataset](#dataset) — and visually overlaps with some recyclable and dry
items), which is the main target for improvement if you retrain this on a
larger/more balanced dataset.

## Open in Colab

[Add your Colab share link here — Colab → Share → "Anyone with the link" → copy]

Runtime: **T4 GPU** (Runtime → Change runtime type → T4 GPU) before running.

## Dataset

Built from two public Kaggle datasets, remapped and merged into the bin's
5-category scheme:

* [`mostafaabla/garbage-classification`](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
  — 12-class general garbage dataset (~15k images: battery, biological,
  brown-glass, cardboard, clothes, green-glass, metal, paper, plastic,
  shoes, trash, white-glass).
* [`engineeringubu/pharmaceutical-and-biomedical-waste`](https://www.kaggle.com/datasets/engineeringubu/pharmaceutical-and-biomedical-waste)
  — biomedical/pharmaceutical waste images, since the general dataset has
  no biomedical category of its own.

Split 80/20 train/validation **within each original class** before
remapping, so the validation set stays representative of each source
category rather than being skewed by the remapping step. Seeded (`seed=42`)
for reproducibility.

## Class mapping

The 12 source classes are consolidated into the bin's 5 operational bins:

| Original class | → Bin |
|---|---|
| biological | **wet** |
| paper, cardboard, clothes, shoes, trash | **dry** |
| plastic | **plastic** |
| batteries + pharmaceutical/biomedical dataset | **biomedical** |
| metal, green-glass, brown-glass, white-glass | **recyclable** |

This mapping is a deliberate design choice, not a limitation of the source
data — e.g. metal and glass are grouped into "recyclable" because that's
how this bin's physical sorting mechanism is organized, and batteries are
grouped with biomedical waste because both need hazardous-material handling
rather than regular disposal.

`class_names.json` (exported alongside the model) preserves the exact
class order used during training — the Flask server and this list must
always stay in sync, since the model's output index has no meaning without
it:

```json
["wet", "dry", "plastic", "biomedical", "recyclable"]
```

## Preprocessing and augmentation

* Images resized to **224×224**, RGB, pixel values rescaled to `[0, 1]`.
* Training-time augmentation (not applied to validation data):
  - Rotation ±20°
  - Width/height shift ±15%
  - Shear 10%, zoom ±20%
  - Horizontal flip
  - **Brightness range 0.6–1.4×** — specifically included because the
    real bin captures images under a flash LED in a chute, which produces
    different lighting than clean dataset photos.
* **Class weights** computed via `sklearn.utils.class_weight.compute_class_weight("balanced", ...)`
  and passed to `model.fit(..., class_weight=...)` in both training phases,
  to counteract the class imbalance visible in the support column above
  (dry and biomedical dominate the dataset; plastic and wet are
  comparatively rare).

## Model architecture

Transfer learning on **MobileNetV2** (ImageNet-pretrained), chosen for:
* Small footprint (~14MB base) — loads quickly on a laptop for real-time
  inference, no GPU required at serving time.
* Strong baseline performance on texture/shape-heavy image classification
  tasks like waste sorting.

```
MobileNetV2 (frozen initially, ImageNet weights, no top)
  → GlobalAveragePooling2D
  → BatchNormalization
  → Dense(256, relu) → Dropout(0.4)
  → Dense(128, relu) → Dropout(0.3)
  → Dense(5, softmax)
```

## Training procedure

Two-phase transfer learning:

**Phase 1 — train the classifier head, base frozen**
* Up to 12 epochs, Adam optimizer, learning rate `1e-3`
* `EarlyStopping` (patience 4, restores best weights) +
  `ReduceLROnPlateau` (factor 0.5, patience 2)

**Phase 2 — fine-tune, last 30 layers of MobileNetV2 unfrozen**
* Up to 20 epochs, Adam optimizer, learning rate dropped to `1e-5`
  (small, since we're adapting pretrained ImageNet features rather than
  learning from scratch)
* `EarlyStopping` (patience 6) + `ReduceLROnPlateau` (factor 0.3,
  patience 3)

Both phases use the same class-weighted loss and the same
train/validation generators.

## Evaluation

After training, the notebook runs the model over the full validation set,
prints a `sklearn.metrics.classification_report` (precision/recall/F1 per
class — reproduced in [Results](#results) above), and plots a confusion
matrix (`confusion_matrix.png`, exported alongside the model).

## How to run this notebook

1. Open in Colab (link above) and switch to a **T4 GPU** runtime.
2. Run all cells top to bottom (`Ctrl+F9`).
3. When prompted, upload your `kaggle.json` API token (from
   [kaggle.com](https://www.kaggle.com) → your profile → Settings → API →
   Create New Token). This authenticates the automatic dataset download —
   no manual dataset upload needed.
4. Training takes roughly 1–1.5 hours total on a T4 (Phase 1 ~5 min/epoch,
   Phase 2 ~4.5 min/epoch, with early stopping usually cutting both phases
   short before the max epoch count).
5. The last cells download three files to your machine:
   `waste_classifier.h5`, `class_names.json`, `confusion_matrix.png`.

## Output files

| File | Purpose |
|---|---|
| `waste_classifier.h5` | Trained model (~26MB), loaded by the Flask server |
| `class_names.json` | Class index → name mapping, must match the model exactly |
| `confusion_matrix.png` | Visual breakdown of per-class errors on the validation set |

## Using the model

Place `waste_classifier.h5` and `class_names.json` in the same folder as
the bin's `laptop_server_final.py`, then run the server. The server:

1. Receives a raw JPEG over HTTP POST at `/classify`.
2. Resizes to 224×224, normalizes to `[0, 1]`.
3. Runs inference, takes the argmax class.
4. Returns that class's index as a single character (`0`–`4`), or `9` if
   confidence is below the server's threshold (default 60%) — the physical
   bin falls back to a moisture sensor in that case rather than risk a
   wrong bin.

See the main project repo's `server/README.md` for the full server setup
and API details.

## Known limitations

* **Plastic** is the weakest-performing class (85.0% precision, fewest
  training examples at 173 validation images) — likely to be the first
  thing worth improving with more/better-labelled data.
* Trained entirely on clean, well-lit dataset photos plus brightness
  augmentation to approximate flash-lit chute conditions — real-world
  accuracy in your specific bin's lighting/camera setup may differ from
  the validation numbers above. Worth spot-checking with actual captured
  images from the ESP32-CAM before trusting the confidence threshold
  as-is.
* The biomedical class is a mix of a dedicated biomedical dataset and
  battery images from the general dataset — these are visually quite
  different object types grouped under one label for operational reasons
  (both need hazardous handling), which may limit how cleanly the model
  can learn a single unified "biomedical" visual concept.

## Possible improvements

* Collect and label real images captured by the deployed ESP32-CAM (under
  actual chute lighting) and fine-tune on those, rather than relying
  solely on generic Kaggle datasets.
* Rebalance or expand the plastic class specifically.
* Export to TensorFlow Lite for on-device inference on the ESP32,
  removing the laptop-server dependency entirely.
* Track per-class confidence distributions in production to tune the
  60% fallback threshold per class rather than globally.

## Author 
AAKASH S S

Dataset licenses: check the original Kaggle dataset pages linked above for
their respective terms before redistributing the data itself (this repo
does not include the raw dataset, only the training code).
