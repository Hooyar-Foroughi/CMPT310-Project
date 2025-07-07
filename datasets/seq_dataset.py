import numpy as np
import torch
from torch.utils.data import Dataset

class SeqSpeakerCount(Dataset):
    def __init__(self, index_df, mapping):
        self.paths   = index_df["npz"].values
        self.targets = index_df["label"].values   # 0..K-1 ints
        self.mapping = mapping                    # list of original counts

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        seq = np.load(self.paths[idx])["seq"]      # (T, 256)
        return torch.from_numpy(seq), self.targets[idx], seq.shape[0]

def collate(batch):
    seqs, labels, lengths = zip(*batch)
    lengths  = torch.tensor(lengths)
    maxT     = lengths.max()
    B        = len(batch)
    emb_dim  = seqs[0].shape[1]
    padded   = torch.zeros(B, maxT, emb_dim)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded, torch.tensor(labels), lengths