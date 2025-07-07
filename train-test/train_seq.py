import pathlib
import sys
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT)) 

from features.seq_dataset import SeqSpeakerCount, collate
from models.seq_count_lstm import SeqCountLSTM


index = pd.read_csv("features/seq_index.csv")
index = index[index["num_speakers"]<=5]
labels_int, class_vals = pd.factorize(index["num_speakers"])
index["label"] = labels_int
num_classes = len(class_vals)

train_df, test_df = train_test_split(index, test_size=0.15,
                                     stratify=index["label"], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.1765,
                                     stratify=train_df["label"], random_state=42)

class_counts = np.bincount(train_df["label"], minlength=num_classes)
sample_w = 1.0 / class_counts[train_df["label"]]
sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_ds = SeqSpeakerCount(train_df, class_vals)
val_ds   = SeqSpeakerCount(val_df,   class_vals)
train_ld = DataLoader(train_ds, batch_size=16, sampler=sampler,
                      collate_fn=collate)
val_ld   = DataLoader(val_ds, batch_size=16, collate_fn=collate)

model = SeqCountLSTM(num_classes=num_classes).to("cpu")
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best, stale, patience = 0, 0, 8
for epoch in range(30):
    model.train()
    for x, y, L in train_ld:
        opt.zero_grad()
        logits = model(x, L)
        loss   = criterion(logits, y)
        loss.backward()
        opt.step()
    # --- val
    model.eval(); correct=tot=0
    with torch.no_grad():
        for x,y,L in val_ld:
            pred = model(x,L).argmax(1)
            correct += (pred==y).sum().item(); tot += y.size(0)
    acc = 100*correct/tot
    print(f"Epoch {epoch+1:02d}  val {acc:4.1f}%")
    if acc>best: best, stale = acc,0; torch.save(model.state_dict(),"models/best_seq.pth")
    else: stale+=1
    if stale>=patience: break