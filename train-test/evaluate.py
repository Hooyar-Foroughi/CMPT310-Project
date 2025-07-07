"""
Evaluate the trained MLP on the held-out test set.

Run from repo root:

    python3 train-test/evaluate.py
"""
import os
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))   

from models.mlp_classifier import SpeakerCountMLP

# ------------------------------------------------------------------
CSV_PATH = os.path.join("features", "labeled_dataset.csv")
df = pd.read_csv(CSV_PATH)

# ----- identical cleaning to train.py -----------------------------
df = df[df["num_speakers"] <= 5]     # keep only files with ≤ 5 speakers
df = df.dropna(subset=["num_speakers"])
vc = df["num_speakers"].value_counts()
df = df[df["num_speakers"].isin(vc[vc > 1].index)]

labels_orig = df["num_speakers"].astype(int)

# remap → contiguous 0..K-1
labels_int, class_values = pd.factorize(labels_orig)   # class_values holds e.g. [2,3,4,5]
df = df.drop(columns=["filename", "num_speakers"])
df = df.select_dtypes(include=[np.number]).fillna(df.mean())
df["num_speakers"] = labels_int

X = df.drop(columns=["num_speakers"]).values
y = df["num_speakers"].values           # contiguous labels

# ------------------------------------------------------------------
# same 70/15/15 stratified split (test = 15 %)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# scale the same way (fit on train+val part)
scaler = StandardScaler()
X_tmp  = scaler.fit_transform(X_tmp)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------------
input_dim    = X_test.shape[1]
num_classes  = len(class_values)
model        = SpeakerCountMLP(input_dim=input_dim, num_classes=num_classes)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

with torch.no_grad():
    logits     = model(torch.tensor(X_test, dtype=torch.float32))
    preds_int  = logits.argmax(1).numpy()

# convert back to real speaker-count values
y_true       = class_values[y_test]
y_pred       = class_values[preds_int]

# ------------------------------------------------------------------
print("Classification Report (speaker counts):")
print(classification_report(y_true, y_pred, digits=3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))