"""
Evaluate the SeqCount-LSTM on the held-out test set.

Run from repo root:

    python3 train-test/eval_seq.py
"""
import os, pathlib, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from features.seq_dataset import SeqSpeakerCount, collate
from models.seq_count_lstm import SeqCountLSTM

# ----------------------------------------------------------------------
IDX_CSV   = REPO_ROOT / "features" / "seq_index.csv"
index     = pd.read_csv(IDX_CSV)

# keep only clips with ≤5 speakers (same as train)
index = index[index["num_speakers"] <= 5]

# remove classes that appear only once
vc = index["num_speakers"].value_counts()
index = index[index["num_speakers"].isin(vc[vc > 1].index)]

# contiguous labels 0..K-1
labels_int, class_vals = pd.factorize(index["num_speakers"].astype(int))
index["label"] = labels_int
num_classes     = len(class_vals)

# 70/15/15 split (same seed as training)
train_df, test_df = train_test_split(
    index, test_size=0.15, stratify=index["label"], random_state=42
)
train_df, val_df  = train_test_split(
    train_df, test_size=0.1765, stratify=train_df["label"], random_state=42
)

# ----------------------------------------------------------------------
test_ds = SeqSpeakerCount(test_df, class_vals)
test_ld = DataLoader(test_ds, batch_size=16, collate_fn=collate)

# model
model = SeqCountLSTM(num_classes=num_classes)
model.load_state_dict(torch.load("models/best_seq.pth", map_location="cpu"))
model.eval()

# ----------------------------------------------------------------------
all_pred, all_true = [], []
with torch.no_grad():
    for x, y, L in test_ld:
        logits = model(x, L)
        all_pred.append(logits.argmax(1).numpy())
        all_true.append(y.numpy())

pred = np.concatenate(all_pred)
true = np.concatenate(all_true)

# map back to real speaker counts
y_true = class_vals[true]
y_pred = class_vals[pred]

# ----------------------------------------------------------------------
print("Classification Report (speaker counts):")
print(classification_report(y_true, y_pred, digits=3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))