"""
evaluate.py
Evaluates the trained model using precision, recall, F1-score, and a
confusion matrix — not just raw accuracy, since accuracy alone can be
misleading for medical classification tasks.

Usage:
    python src/evaluate.py
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from train import train_and_save


def evaluate():
    model, scaler, X_test, y_test, is_synthetic = train_and_save()

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    if is_synthetic:
        print("NOTE: Trained on SYNTHETIC demo data. These numbers are")
        print("illustrative of the pipeline only, not real performance.")
        print("-" * 50)

    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print("               Predicted No   Predicted Yes")
    print(f"Actual No      {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual Yes     {cm[1][0]:<14} {cm[1][1]}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    evaluate()
