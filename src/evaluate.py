"""
Evaluation helpers.

Usage
-----
from src.evaluate import evaluate_dataframe

df = pd.DataFrame([...])  # must have columns: filename, true, pred
evaluate_dataframe(df, out_prefix="results/cluster_spectral")

This will:
* Compute classification report, confusion matrix, overall accuracy.
* Save CSV and TXT files under the given prefix:
    cluster_spectral.csv
    cluster_spectral.txt
and return the report string for optional printing.
"""

from __future__ import annotations
import os
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

__all__ = ["evaluate_dataframe"]

def evaluate_dataframe(df, out_prefix="results/eval", digits=3, zero_division=0):
    required = {"true", "pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {required}")

    y_true = df["true"].astype(int)
    y_pred = df["pred"].astype(int)

    report = classification_report(
        y_true, y_pred, digits=digits, zero_division=zero_division
    )
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # ---- save outputs -------------------------------------------------
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    df.to_csv(f"{out_prefix}.csv", index=False)

    with open(f"{out_prefix}.txt", "w") as fh:
        fh.write("Classification Report:\n")
        fh.write(report)
        fh.write("\nConfusion Matrix:\n")
        fh.write(str(cm))
        fh.write(f"\nOverall Accuracy: {acc * 100:.2f}%\n")

    return report