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

# Waste material classifier (course project)

Upload a photo of one item. The model returns its top-3 guesses among glass, metal, paper and plastic,
and a Grad-CAM heat map showing which part of the photo drove the guess.

- Trained on TrashNet (Thung and Yang, 2016, MIT licence): single objects on a white background.
- Course project, originally DSAI4101 Deep Learning (UDST, Fall 2025), rebuilt in 2026.
- It is not reliable on real bins or sorting lines. Not for real sorting decisions.
- Code, data split and test results: see the GitHub repository linked from the author's profile.
