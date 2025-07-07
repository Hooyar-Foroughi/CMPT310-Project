import torch, torch.nn as nn, torch.nn.functional as F

class SeqCountLSTM(nn.Module):
    def __init__(self, emb_dim=256, hidden=128, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, hidden,
                            bidirectional=True,
                            batch_first=True)
        self.attn = nn.Linear(hidden*2, 1)   # energy score
        self.fc   = nn.Linear(hidden*2, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # attention weights
        alpha = F.softmax(self.attn(out).squeeze(-1), dim=1)   # (B, T)
        pooled = (out * alpha.unsqueeze(-1)).sum(1)            # (B, 2H)
        return self.fc(pooled)