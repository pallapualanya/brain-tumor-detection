"""
train.py
Trains a classifier to detect brain tumors from MRI images.

Usage:
    python src/train.py

Uses data/yes and data/no if present, otherwise falls back to synthetic
demo data so you can see the full pipeline run end-to-end.

Model: MLPClassifier (a fully-connected neural network) from scikit-learn.
For a real convolutional neural network (CNN) suited to production-grade
image classification, see train_cnn_tensorflow.py (requires TensorFlow).
"""

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataset


def train_and_save(model_path="brain_tumor_model.joblib"):
    (X, y), is_synthetic = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    joblib.dump({"model": model, "scaler": scaler}, model_path)
    print(f"Model saved to {model_path}")

    return model, scaler, X_test_scaled, y_test, is_synthetic


if __name__ == "__main__":
    train_and_save()
