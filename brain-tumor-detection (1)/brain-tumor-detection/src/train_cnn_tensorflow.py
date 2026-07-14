"""
train_cnn_tensorflow.py
OPTIONAL upgrade: a real Convolutional Neural Network (CNN) using
TensorFlow/Keras, for genuine deep-learning image classification.

This script requires TensorFlow, which is NOT installed in every
environment. Install it first:
    pip install tensorflow

Note: this script was written carefully following standard Keras patterns
but has not been execution-tested in this environment (no TensorFlow
available here). Test it locally or on Google Colab before relying on it.

Usage:
    python src/train_cnn_tensorflow.py
"""

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    raise ImportError(
        "TensorFlow is required for this script. Install with: pip install tensorflow"
    )

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_SIZE = 128


def load_images_as_arrays(data_dir="data"):
    """Loads data/yes and data/no folders as image arrays for the CNN."""
    X, y = [], []
    for label, folder in [(1, "yes"), (0, "no")]:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = tf.keras.utils.load_img(
                img_path, color_mode="grayscale", target_size=(IMAGE_SIZE, IMAGE_SIZE)
            )
            arr = tf.keras.utils.img_to_array(img) / 255.0
            X.append(arr)
            y.append(label)

    if not X:
        raise FileNotFoundError(
            f"No images found under '{data_dir}/yes' and '{data_dir}/no'. "
            "Download a real MRI dataset first, e.g.: "
            "https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection"
        )

    return np.array(X), np.array(y)


def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train():
    X, y = load_images_as_arrays()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_cnn()
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=16,
    )

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

    model.save("brain_tumor_cnn.keras")
    print("Model saved to brain_tumor_cnn.keras")

    return model, history


if __name__ == "__main__":
    train()
