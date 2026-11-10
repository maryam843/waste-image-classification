# Publishing the demo on Hugging Face Spaces (done by hand)

The model file is never committed to GitHub. You create it on Colab and upload it yourself.

Live Space: https://huggingface.co/spaces/Maryamdafc/waste-image-classification (private for now).

## Things the Space needs (as of October 2026)

- **Hardware: ZeroGPU.** Free Gradio Spaces run on ZeroGPU; the CPU basic option needs a PRO account.
- **Python 3.10.** The Space runs Python 3.10, so `app.py`, `models.py` and `gradcam.py` must stay 3.10-compatible
  (CI lints them with ruff's Python 3.10 target).
- **A ZeroGPU-supported torch version.** `hf_space/requirements.txt` pins `torch==2.11.0` and `torchvision==0.26.0`
  (the versions the Colab run used). These differ on purpose from the repo's `requirements.txt`, which is for local
  runs and CI. Check the ZeroGPU documentation before changing them.
- **No gradio line in requirements.txt.** The Space installs Gradio itself from `sdk_version` in the Space card
  (`hf_space/README.md`).
- **`@spaces.GPU` on the inference function.** `classify` in `app.py` carries it. `import spaces` is wrapped in
  try/except with a no-op fallback, so the same file runs locally and in tests without the `spaces` package.

## Steps

1. Run `notebooks/waste_classification.ipynb` on Colab (full config). The last cells write
   `exports/demo_model.pt` and build the folder `hf_space_build/` with:
   `app.py`, `requirements.txt`, `README.md`, `models.py`, `gradcam.py`, `model.pt`.
   Download that folder (the notebook zips it as `hf_space_build.zip`).
   Locally you can build the same folder with:
   `python scripts/prepare_hf_space.py --model exports/demo_model.pt --out hf_space_build`
2. On huggingface.co: New -> Space. Name it, SDK = Gradio, hardware = ZeroGPU (free),
   visibility = your choice (private first is fine).
3. In the Space: Files -> Add file -> Upload files. Upload all six files from `hf_space_build/`
   (the README.md there is the Space card; keep its YAML header).
4. Wait for the build log to show "Running". Upload one TrashNet-style photo and check that you get
   three classes with probabilities and a Grad-CAM image.
5. If you change `app.py` or `requirements.txt` on the Space directly, copy the same change into `hf_space/`
   in this repo so the two stay identical.

Do not upload any token or the dataset. If you ever paste a Hugging Face token anywhere by mistake,
revoke it at huggingface.co -> Settings -> Access Tokens.
