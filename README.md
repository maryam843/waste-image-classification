# Waste image classification: why a CNN beats a plain DNN on images

A fair, reproducible comparison of my 2025 fully connected network against a small CNN and a frozen ResNet-18
on public TrashNet photos of glass, metal, paper and plastic, with Grad-CAM explanations and a small demo app.

Originally built for DSAI4101 Deep Learning, UDST, Fall 2025. Rebuilt on public data in 2026.
My full transfer-learning work is in my [OryxEye](https://github.com/maryam843/OryxEye) project.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maryam843/waste-image-classification/blob/main/notebooks/waste_classification_2026.ipynb)

## In plain English

- **What it is:** a program that looks at a photo of one piece of rubbish and says whether it is glass, metal, paper or plastic.
- **Who it helps:** recycling sorters and "smart bin" projects, where sorting by hand is slow and mistakes contaminate whole batches.
- **How it works:** photo -> resized to 224x224 -> a neural network -> the most likely material, plus a heat map (Grad-CAM) of the part of the photo that decided it.
- **2025 vs 2026:** in 2025 I trained one fully connected network for a course assignment. In 2026 I rebuilt it on public data, fixed how it was evaluated, and compared it with a CNN and a pretrained ResNet-18.
- **Honest result:** TODO after the Colab run (numbers will come only from `results/results.json`). The 2025 and 2026 scores use different data and cannot be compared directly.
- **Why it matters here:** Qatar's national waste strategy aims to raise recycling, and automated sorting is part of that. This project shows the core skills: building, evaluating and explaining an image model honestly.

## The problem

Mixed recyclables have to be separated by material before they can be recycled. Most of this is still done by
people on sorting lines. In Qatar, recycling targets under the national waste strategy and the Qatar National Vision 2030
environmental pillar make automated sorting and smart bins a practical use case for computer vision.

This repo does not build a sorting system. It answers a narrower question that I got wrong in 2025: for small image
datasets, how much does the model type matter, when everything is evaluated fairly?

## 2025 vs 2026

| | 2025 coursework | 2026 rebuild |
|---|---|---|
| Data | Course `data.zip`, 1,540 train / 100 test images (no longer available) | Public TrashNet, 1,981 images after cleaning, 70/15/15 split |
| Models | Fully connected DNN [512,256,128] | Same DNN, plus SmallCNN and frozen ResNet-18, plus 2 baselines |
| Model selection | On the test set | On a validation set; test used once per trained model |
| Seeds | One run | 3 seeds per model, mean +/- sd, 95 % Wilson intervals |
| Reproducible | No (cells ran out of order, data missing) | Yes: pinned data hash, committed split, tests and CI |
| Explainability | Misclassified images | Grad-CAM for SmallCNN and ResNet-18 |

The original notebook is kept as submitted (personal data removed) in
[`notebooks/original_2025_dnn.ipynb`](notebooks/original_2025_dnn.ipynb). Its reported test F1 was 0.6988 on 100 images,
but that number was selected on the same test set, so it is optimistic.

## Data

- **Source:** [TrashNet](https://github.com/garythung/trashnet) by Gary Thung and Mindy Yang (Stanford CS 229, 2016),
  MIT licence. The images are downloaded by the code from the TrashNet repository at a pinned commit and checked against a
  SHA-256. They are not redistributed here.
- **Classes used:** glass (501), metal (410), paper (594), plastic (482). Cardboard and trash are not used, to match the 2025 task.
- **Cleaning:** every image is hashed. Three files appear twice with different labels (for example the same photo in both
  `glass/` and `plastic/`), so those 6 files are dropped. That leaves 1,981 images.
- **Split:** stratified 70/15/15 on unique image hashes with seed 42, so exact duplicates can never be split across
  train and test: 1,386 train, 297 validation, 298 test. The split is saved as a CSV of hashes (no images) in
  [`data_split/trashnet_split.csv`](data_split/trashnet_split.csv).
- **Not comparable with 2025:** the course data is no longer available, so 2025 and 2026 scores are not directly comparable.

## Method

All models see the same 224x224 images, normalised with ImageNet statistics, and are trained with Adam (learning rate 1e-3),
cross-entropy, ReduceLROnPlateau and early stopping. The learning-rate schedule, early stopping and the choice of checkpoint
all use validation macro F1. Each model is trained with 3 seeds; each trained model is scored on the test set exactly once.

| Model | What it is | Why it is here |
|---|---|---|
| Majority class | Always predicts the most common training class | Floor |
| Logistic regression | Linear model on 64x64 pixels, C chosen on validation | Is a neural network needed at all? |
| DNN [512,256,128] | My 2025 network, unchanged: flatten, 3 hidden layers with BatchNorm and Dropout 0.5 (77,237,380 parameters) | The 2025 approach, trained properly |
| SmallCNN | 4 conv blocks with BatchNorm, global average pooling, trained from scratch with light augmentation | Does using image structure help? |
| ResNet-18 (frozen) | ImageNet weights, only a new linear head is trained | Reference for transfer learning |

Design choices and the reasons for each are in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Results vs baselines

<!-- RESULTS:START -->
TODO: filled in from `results/results.json` after I run `notebooks/waste_classification_2026.ipynb` on Colab.
No numbers are shown until then.
<!-- RESULTS:END -->

Figures (created by the notebook, TODO until the Colab run): learning curves, confusion matrices, misclassified test images
sampled across all four classes, Grad-CAM, and a results bar chart, all in [`figures/`](figures/).

**Grad-CAM:** TODO, `figures/gradcam.png` appears here after the Colab run.

**Live demo:** TODO, Hugging Face Space link after I upload it (steps in [`hf_space/INSTRUCTIONS.md`](hf_space/INSTRUCTIONS.md)).

**Demo video:** TODO, I will record a short clip of the app myself.

## What went wrong in 2025 and how I fixed it

1. **I picked the model using the test set.** Early stopping, the learning-rate schedule, the saved checkpoint and my
   hyperparameter search all looked at the 100 test images. Fix: a separate validation split; `train_model()` does not even
   accept a test loader, and a test checks that the test set is used exactly once per model.
2. **My notebook cells ran out of order.** The "60-epoch experiment" actually continued training the already trained model.
   Fix: every run builds a fresh model from a factory, and CI runs the new notebook top to bottom.
3. **My write-up named the wrong final model.** I wrote [1024,512,256,128], but the reported model was [512,256,128].
   Fix: the architecture and parameter count in `results.json` are read from the trained model itself.
4. **A fully connected network ignores image layout.** Flattening a 224x224 photo gives 150,528 inputs, and 99.8 % of the
   77 M parameters sit in the first layer. Fix: compare it against a CNN and a pretrained ResNet-18 under the same rules.
5. **The course data is gone.** Fix: public TrashNet with a pinned hash and a committed split, so anyone can rebuild the
   exact same sets.

## Limitations

- TrashNet photos are single, clean objects on a white background under good light. They are not photos from real bins or
  sorting lines, so these scores say nothing about real-world sorting.
- Only 4 classes; cardboard, trash, food waste and mixed items are not covered.
- About 300 test images: differences of a few images are within noise, which is why I report 3 seeds and Wilson intervals.
- The ResNet-18 backbone is frozen; I did not fine-tune it here (that is in OryxEye).
- The demo is a course project. It is not for real sorting decisions.

## What I learned

- A good-looking number means little if the test set helped choose the model. I now keep a validation set for every decision.
- Notebooks hide state. Moving the logic into `src/` with tests made out-of-order runs impossible.
- For images, architecture matters more than tuning: my 2025 hyperparameter search could not fix a model that ignores pixel layout.
- Reporting mean and spread over seeds, plus a confidence interval, stops me from reading noise as progress.
- Checking data is part of the model: hashing TrashNet found duplicates with conflicting labels that I would have missed.
- Writing down the reason for each choice (`docs/DECISIONS.md`) made it much easier to explain my work.

## How to run

**On Colab (full run, GPU):** open the notebook with the badge above, choose Runtime > Change runtime type > T4 GPU,
then Runtime > Run all. While this repo is private, add a read-only GitHub token as a Colab secret named `GH_TOKEN` so the
notebook can clone the code (or upload the repo yourself). The notebook writes `results/results.json`, `figures/`, and the
demo model (not committed), and downloads them as zip files.

**Locally on CPU (smoke run on tiny synthetic images, about a minute):**

<!-- readme-commands -->
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu   # Linux, CPU only
pip install -r requirements-dev.txt
pip install -e .
pytest
python -m waste_cls.cli run --config smoke --out runs/smoke
python scripts/pii_scan.py
python scripts/check_readme_claims.py
```

`python -m waste_cls.cli prepare-data` downloads TrashNet (about 43 MB) and checks that the rebuilt split matches the
committed CSV. `python -m waste_cls.cli run --config full --out results --figures figures --export exports/demo_model.pt`
is the full run (use a GPU). `python -m waste_cls.cli predict --model exports/demo_model.pt --image photo.jpg --cam cam.png`
gives the top-3 classes and a Grad-CAM image for one photo.

## Use of AI tools

In 2025 I used AI help for the hyperparameter tuning loops and for setting up the environment, and I said so in the
original notebook. For the 2026 rebuild I also used an AI coding assistant to help restructure the code into `src/`,
write tests and CI, and draft documentation. I reviewed the code and the design choices, and every number in this README
has to come from `results/results.json`, which a script checks.

## Credits

- TrashNet dataset: Gary Thung and Mindy Yang, "Classification of Trash for Recyclability Status", Stanford CS 229, 2016.
  [github.com/garythung/trashnet](https://github.com/garythung/trashnet), MIT licence.
- ResNet-18: He et al., 2016, ImageNet weights from torchvision.
- Grad-CAM: Selvaraju et al., 2017.
- Course: DSAI4101 Deep Learning, University of Doha for Science and Technology.

## Licence

My code is under the MIT licence (see [`LICENSE`](LICENSE)). TrashNet images are not included; they keep their own MIT licence.
