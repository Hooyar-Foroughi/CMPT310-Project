"""
Train MLP classifier to predict speaker count.
Run from repo root:

    python train-test/train.py
"""

import os
import sys
import json
import torch
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import WeightedRandomSampler

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))        

from models.mlp_classifier import SpeakerCountMLP

DATA_CSV = os.path.join("features", "labeled_dataset.csv")
df = pd.read_csv(DATA_CSV)

# ---- clean & prepare -------------------------------------------------
df = df[df["num_speakers"] <= 5]     # keep only files with ≤ 5 speakers
df = df.dropna(subset=["num_speakers"])
vc = df["num_speakers"].value_counts()
df = df[df["num_speakers"].isin(vc[vc > 1].index)]

# map speaker-count → contiguous class index 0..K-1
labels_int, class_values = pd.factorize(df["num_speakers"].astype(int))
num_classes = len(class_values)

# keep only numeric feature columns (drop filename + label)
df = df.drop(columns=["filename", "num_speakers"])
df = df.select_dtypes(include=[np.number]).fillna(df.mean())

# final feature matrix & target vector
X = df.values                    # numpy array, shape (N, F)
y = labels_int                   # contiguous ints 0..K-1

# ------------------------------------------------------------------
# split 70/15/15 stratified
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=42
)  # 0.1765×0.85 ≈ 0.15

# -------- balanced sampler *on training set only* ------------
class_counts = np.bincount(y_train, minlength=num_classes)      # counts in train split
sample_weights = 1.0 / class_counts[y_train]                    # one weight per train clip
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(y_train),    # must equal len(train_ds)
    replacement=True
)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ------------------------------------------------------------------
# torch datasets
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_ds = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=64)

# ------------------------------------------------------------------
model = SpeakerCountMLP(input_dim=X_train.shape[1], num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
NUM_EPOCHS = 20

best_val = 0
patience = 8     # stop after 4 epochs with no improvement
stale    = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # ----- train ----------------------------
    model.train()
    tot_loss = tot_correct = tot_samples = 0
    for xb, yb in train_loader:
        xb = xb + 0.01 * torch.randn_like(xb)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        tot_loss   += loss.item()
        tot_correct += (logits.argmax(1) == yb).sum().item()
        tot_samples += yb.size(0)

    train_acc = 100 * tot_correct / tot_samples

    # ----- validation -----------------------
    model.eval()
    val_correct = val_samples = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb).argmax(1)
            val_correct += (preds == yb).sum().item()
            val_samples += yb.size(0)

    val_acc = 100 * val_correct / val_samples
    print(f"Epoch {epoch:02d}: train-acc {train_acc:5.1f}%   val-acc {val_acc:5.1f}%")
    
    if val_acc > best_val:
        best_val = val_acc
        stale = 0
        torch.save(model.state_dict(), "models/best_model.pth")
    else:
        stale += 1
        if stale >= patience:
            print("Early stopping.")
            break


os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
print("\nModel saved →  models/model.pth")

mapping_path = "models/label_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(
        {int(idx): int(v) for idx, v in enumerate(class_values.astype(int))},
        f, indent=2
    )
print(f"Label mapping saved → {mapping_path}")