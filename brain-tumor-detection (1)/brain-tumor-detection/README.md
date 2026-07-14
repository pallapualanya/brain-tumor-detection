# Brain Tumor Detection

A deep learning model that classifies brain tumors from medical imaging data, achieving 90%+ accuracy, with performance evaluated through precision, recall, F1-score, and confusion matrix analysis.

## Overview

This project applies deep learning to a real diagnostic challenge — classifying brain tumors from imaging data. Rather than stopping at accuracy alone, the model's performance was evaluated using multiple metrics to get a fuller, more honest picture of how well it actually performs, especially important in a medical context where false negatives/positives carry real consequences.

## Features

- **Deep learning classification model** for brain tumor detection
- **90%+ classification accuracy** on the evaluation set
- **Multi-metric evaluation**: precision, recall, F1-score, and confusion matrix — not just raw accuracy
- **Structured documentation** of methodology and findings

## Tech Stack

- **Language:** Python
- **Deep Learning:** CNN-based architecture (Convolutional Neural Network) for image classification
- **Evaluation:** scikit-learn metrics (precision, recall, F1-score, confusion matrix)

## How It Works

1. Medical imaging data (e.g., MRI scans) is preprocessed and prepared for training
2. A deep learning model (CNN) is trained to distinguish between tumor and non-tumor images (or between tumor types, depending on dataset labels)
3. The trained model is evaluated on a held-out test set
4. Performance is assessed beyond just accuracy — using precision, recall, F1-score, and a confusion matrix to understand where the model succeeds and where it struggles

## Why Multiple Metrics Matter

Accuracy alone can be misleading, especially with imbalanced medical datasets. This project deliberately evaluated:
- **Precision** — of the tumors flagged, how many were correct
- **Recall** — of all actual tumors, how many were caught
- **F1-score** — the balance between precision and recall
- **Confusion matrix** — a full breakdown of correct vs. incorrect predictions by class

## What I Learned

- Building and training a CNN for image classification tasks
- The importance of evaluating models with more than one metric, especially in high-stakes domains like healthcare
- Interpreting a confusion matrix to identify specific weaknesses in a model's predictions
- Documenting technical findings in a clear, structured way

## Possible Future Improvements

- Expand the dataset for better generalization
- Experiment with transfer learning using pre-trained medical imaging models
- Add Grad-CAM or similar visualization to show which parts of an image influenced the model's decision
- Deploy the model behind a simple web interface for demonstration purposes

## Author

Alanya Pallapu — Computer Science Engineering, MRCET, Hyderabad
