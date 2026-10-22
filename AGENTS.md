# AGENTS.md

## Overview
Fruit image recommendation / retrieval system (university database-of-multimedia project): extracts image features (HOG/FFT/skimage descriptors), stores images in MongoDB (GridFS), finds similar fruits by cosine similarity, and fine-tunes a Vision Transformer (ViT) classifier.

## Layout
- `Code.ipynb` — dataset download (kagglehub), MongoDB/GridFS storage experiments.
- `Code2.ipynb` — feature extraction + similarity search (scikit-image, scipy, sklearn).
- `train_vit.py` — fine-tunes `google/vit-base-patch16-224-in21k` on a Fruits-360 ImageFolder via Hugging Face `Trainer`.
- `DataLoc/<fruit class>/` — small sample of Fruits-360 (100x100 px) images, one folder per class.

## Commands
- Notebooks: `jupyter notebook`
- Train ViT: `python train_vit.py` (edit `data_dir` first — it points to `/home/ubuntu/1/BTL_CSDLDPT/...` on the original machine).
- Needs a local MongoDB (`mongodb://localhost:27017/`) for the GridFS notebook.

## Rules
- Do not commit the full Fruits-360 dataset, model checkpoints or `.bash_history` (gitignored).
- Keep class folder names in `DataLoc/` unchanged (they are the labels).
- Replace hard-coded absolute paths with arguments/relative paths when editing; never hard-code credentials (use env vars).
