# Publishing the demo on Hugging Face Spaces (done by hand)

The model file is never committed to GitHub. You create it on Colab and upload it yourself.

1. Run `notebooks/waste_classification_2026.ipynb` on Colab (full config). The last cells write
   `exports/demo_model.pt` and build the folder `hf_space_build/` with:
   `app.py`, `requirements.txt`, `README.md`, `models.py`, `gradcam.py`, `model.pt`.
   Download that folder (the notebook zips it as `hf_space_build.zip`).
   Locally you can build the same folder with:
   `python scripts/prepare_hf_space.py --model exports/demo_model.pt --out hf_space_build`
2. On huggingface.co: New -> Space. Name it (for example `waste-material-classifier`), SDK = Gradio,
   hardware = CPU basic (free), visibility = your choice (private first is fine).
3. In the Space: Files -> Add file -> Upload files. Upload all six files from `hf_space_build/`
   (the README.md there is the Space card; keep its YAML header).
4. Wait for the build log to show "Running". Upload one TrashNet-style photo and check that you get
   three classes with probabilities and a Grad-CAM image.
5. Copy the Space URL into README.md (section "Live demo"), replacing the TODO, and push.

Do not upload any token or the dataset. If you ever paste a Hugging Face token anywhere by mistake,
revoke it at huggingface.co -> Settings -> Access Tokens.
