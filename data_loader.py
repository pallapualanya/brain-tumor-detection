"""
data_loader.py
Loads brain MRI images for tumor classification.

Expected folder structure (standard Kaggle "Brain MRI Images for Brain Tumor Detection" layout):

    data/
        yes/    <- MRI images WITH a tumor
        no/     <- MRI images WITHOUT a tumor

Download a real dataset (recommended, free) from:
    https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

If no dataset is present, this module can generate synthetic placeholder
data so the rest of the pipeline (training, evaluation) can still be
demonstrated end-to-end.
"""

import os
import numpy as np

IMAGE_SIZE = 64  # images are resized to IMAGE_SIZE x IMAGE_SIZE


def _try_import_pillow():
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


def load_real_dataset(data_dir="data"):
    """
    Loads images from data/yes and data/no folders.
    Returns (X, y) where X is a flattened, normalized pixel array
    and y is 1 for tumor, 0 for no tumor.

    Requires Pillow: pip install pillow
    """
    Image = _try_import_pillow()
    if Image is None:
        raise ImportError("Pillow is required to load real images. Install with: pip install pillow")

    X, y = [], []
    for label, folder in [(1, "yes"), (0, "no")]:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            try:
                img = Image.open(filepath).convert("L")  # grayscale
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(label)
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

    if not X:
        raise FileNotFoundError(
            f"No images found under '{data_dir}/yes' and '{data_dir}/no'. "
            "Download a dataset first (see module docstring)."
        )

    return np.array(X), np.array(y)


def generate_synthetic_dataset(n_samples=400, seed=42):
    """
    Generates synthetic 'MRI-like' data purely so the full pipeline
    (train -> evaluate -> predict) can be demonstrated without a real
    dataset. This is NOT medical data and produces NO real diagnostic value.
    Replace with load_real_dataset() for actual use.
    """
    rng = np.random.default_rng(seed)
    n_features = IMAGE_SIZE * IMAGE_SIZE

    # Class 0 ("no tumor"): lower-intensity, low-variance noise pattern
    X0 = rng.normal(loc=0.35, scale=0.08, size=(n_samples // 2, n_features))

    # Class 1 ("tumor"): higher-intensity core region + noise, to simulate
    # a distinguishable bright mass somewhere in the image
    X1 = rng.normal(loc=0.35, scale=0.08, size=(n_samples // 2, n_features))
    # inject a brighter "tumor" patch into a random contiguous block per sample
    for i in range(X1.shape[0]):
        start = rng.integers(0, n_features - 200)
        X1[i, start:start + 200] += rng.normal(loc=0.35, scale=0.05, size=200)

    X = np.vstack([X0, X1])
    X = np.clip(X, 0, 1)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def load_dataset(data_dir="data", use_synthetic_fallback=True):
    """
    Tries to load the real dataset first; falls back to synthetic data
    (with a clear warning) if none is found and fallback is allowed.
    """
    try:
        return load_real_dataset(data_dir), False
    except (FileNotFoundError, ImportError) as e:
        if use_synthetic_fallback:
            print(f"[data_loader] Real dataset not found ({e}).")
            print("[data_loader] Falling back to SYNTHETIC demo data. "
                  "Results will NOT reflect real diagnostic performance.")
            return generate_synthetic_dataset(), True
        raise
