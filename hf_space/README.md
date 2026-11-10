---
title: Waste Material Classifier
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
pinned: false
license: mit
---

# Waste material classifier

Upload a photo of one item. The model returns its top-3 guesses among glass, metal, paper and plastic,
and a Grad-CAM heat map showing which part of the photo drove the guess.

- Trained on TrashNet (Thung and Yang, 2016, MIT licence): single objects on a white background.
- Model: ImageNet ResNet-18 with a frozen backbone and a trained linear head, chosen on validation macro F1.
- It is not reliable on real bins or sorting lines. Not for real sorting decisions.
- Code, data split and test results: https://github.com/maryam843/waste-image-classification
