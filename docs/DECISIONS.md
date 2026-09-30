# Design decisions

One reason per choice. If a choice changes, the reason changes here too.

## Data

- **Dataset: TrashNet.** It is public, MIT-licensed, has the same four classes as my 2025 course data
  (glass, metal, paper, plastic) and similar photos (one object on a plain background). The course `data.zip` is no longer
  available, so a public set is the only way to make the study reproducible.
- **Download source: TrashNet GitHub repository at a pinned commit, with a SHA-256 check.** The pinned commit and hash
  guarantee the same bytes every time. The Hugging Face copy is what the TrashNet README links to, but I could not verify
  it is byte-identical, so I use the repository copy.
- **Only 4 of the 6 classes.** Keeps the task the same as 2025 so the story (same network, fair evaluation) holds.
- **Hash every image; drop duplicates with conflicting labels.** Three photos appear in two class folders. Their true label
  is unknown, so keeping either copy would add label noise; 6 files out of 1,987 are dropped and listed in `results.json`.
- **Split: stratified 70/15/15 on unique hashes, seed 42.** Stratified so every split has the same class mix; on hashes so
  exact duplicates can never leak between train and test; 15 % validation is enough (297 images) for early stopping and
  model choice while keeping 298 test images.
- **Split committed as a CSV of hashes.** Anyone can rebuild the identical split without me redistributing images, and the
  code refuses to run if the rebuilt split differs from the CSV.

## Preprocessing

- **Image size 224x224 for every network.** Same as 2025 (so the DNN is exactly the 2025 network with 150,528 inputs) and
  the native input size for ResNet-18. TrashNet photos are 512x384, so this squashes the aspect ratio a little, which is
  also what the 2025 notebook did.
- **ImageNet mean/std normalisation for every model.** ResNet-18 needs it; using it for all models keeps inputs identical.
- **Images cached as uint8 tensors in memory.** 1,981 images at 224x224 fit in about 300 MB, which makes each epoch fast
  and keeps augmentation deterministic per seed.
- **Augmentation: horizontal flip, rotation up to 10 degrees (white fill), colour jitter 0.2.** Light changes that keep
  the material recognisable; white fill matches the white TrashNet background. Used for SmallCNN and ResNet-18.
- **No augmentation for the DNN.** The point is to test the 2025 network as it was, trained properly; adding augmentation
  would change two things at once.

## Models

- **DNN [512,256,128], BatchNorm, Dropout 0.5.** Exactly the network behind the 2025 F1 0.6988 (not the [1024,...] one my
  write-up wrongly named). 77,237,380 parameters.
- **SmallCNN with 4 conv blocks (32-64-128-256), BatchNorm, global average pooling, dropout 0.3.** Small enough to train
  from scratch on 1,386 images in minutes, with about 0.4 M parameters (under 1 % of the DNN), so any gain comes from using
  image structure, not from size.
- **ResNet-18 with frozen ImageNet backbone and a linear head.** The cheapest honest transfer-learning reference. The
  backbone is kept in eval mode so its BatchNorm statistics do not drift. Fine-tuning is in OryxEye, not duplicated here.
- **Baselines: majority class and logistic regression on 64x64 pixels.** Majority is the floor; logistic regression shows
  what a linear model on raw pixels gets, which the DNN must beat to justify itself. 64x64 keeps it fast.

## Training and evaluation

- **Adam, learning rate 1e-3, batch 32.** Same as 2025, so the only change for the DNN is the evaluation protocol.
- **ReduceLROnPlateau (factor 0.5, patience 3) on validation macro F1.** 2025 stepped the scheduler on test F1; now it
  only sees validation.
- **Early stopping on validation macro F1 (patience 8, max 40 epochs), keep the best-validation weights.** Stops
  overfitting without looking at the test set.
- **Macro F1 as the selection metric.** The classes are not balanced (paper 594 vs metal 410), and macro F1 weights each
  class equally.
- **3 seeds per network (0, 1, 2).** One run can be lucky; mean and sd over seeds show how stable a result is.
- **Test set evaluated once per trained model.** Enforced in code (`run_single` is the only place the test loader exists)
  and by a regression test.
- **95 % Wilson interval for accuracy.** With about 300 test images the uncertainty is a few percentage points; Wilson is
  accurate for proportions near 0 or 1 where the normal approximation is not.
- **Deterministic seeding (Python, NumPy, torch, DataLoader generator), cudnn deterministic.** Same seed gives the same
  split and the same first-epoch loss on CPU (tested).

## Explainability and demo

- **Grad-CAM on the last conv layer of SmallCNN and on `layer4[-1]` of ResNet-18.** The last conv layer has the most
  semantic features while keeping some spatial resolution (28x28 for SmallCNN, 7x7 for ResNet-18 at 224 input).
- **Grad-CAM images spread across all 4 classes, errors sampled per class.** 2025 showed the first 8 errors, which were
  all glass.
- **Demo model = single run with the best validation macro F1 among Grad-CAM capable models.** Chosen on validation, never
  on test, and it must support Grad-CAM for the demo.
- **Gradio on Hugging Face Spaces, free CPU.** Free, no server to maintain, and recruiters can click it. I upload it by
  hand so no token ever touches this repo.
- **Model files saved as plain tensors and loaded with `weights_only=True`.** Loading never unpickles arbitrary code.

## Engineering

- **Code in `src/waste_cls/`, notebook only calls it.** Notebook cells cannot hide state that the tests do not see.
- **Smoke config on tiny synthetic images for CI.** CI runs the real code paths (including the notebook) on CPU in about a
  minute, with no downloads.
- **Pinned requirements, Python 3.12.** Reproducible installs; Colab uses its own preinstalled versions, which are
  recorded in `results.json`.
