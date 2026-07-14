"""
predict.py
Loads a trained model and predicts whether a single MRI image shows a tumor.

Usage:
    python src/predict.py path/to/image.jpg
"""

import sys
import joblib
import numpy as np

from data_loader import IMAGE_SIZE, _try_import_pillow


def predict_image(image_path, model_path="brain_tumor_model.joblib"):
    Image = _try_import_pillow()
    if Image is None:
        raise ImportError("Pillow is required. Install with: pip install pillow")

    saved = joblib.load(model_path)
    model, scaler = saved["model"], saved["scaler"]

    img = Image.open(image_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32).flatten() / 255.0
    arr_scaled = scaler.transform([arr])

    pred = model.predict(arr_scaled)[0]
    proba = model.predict_proba(arr_scaled)[0]

    label = "TUMOR DETECTED" if pred == 1 else "NO TUMOR DETECTED"
    confidence = proba[pred] * 100

    print(f"Result: {label}")
    print(f"Confidence: {confidence:.2f}%")
    return label, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)
    predict_image(sys.argv[1])
