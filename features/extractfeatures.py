"""
Extract Resemblyzer speaker-embeddings for every clip.

Outputs
-------
1. features/labeled_dataset.csv     (pooled stats + k_elbow)
2. features/seq_index.csv           (filename, npz path, label)
3. features/seq/<fname>.npz         (partial embeddings array)

Run from repo root:
    python3 features/extractfeatures.py
"""
import os, sys, pathlib, warnings, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from resemblyzer import VoiceEncoder, preprocess_wav

# ----------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))          # import utils

from utils.label_utils import count_unique_speakers

WAV_DIR  = REPO_ROOT / "data" / "wav"
RTTM_DIR = REPO_ROOT / "data" / "rttm"

STATS_CSV = REPO_ROOT / "features" / "labeled_dataset.csv"
SEQ_CSV   = REPO_ROOT / "features" / "seq_index.csv"
SEQ_DIR   = REPO_ROOT / "features" / "seq"

for d in (STATS_CSV.parent, SEQ_DIR):
    d.mkdir(parents=True, exist_ok=True)

encoder = VoiceEncoder()
rows_stats, rows_seq = [], []

warnings.filterwarnings("ignore", category=FutureWarning)   # mute sklearn msg

# ----------------------------------------------------------------------
for fname in os.listdir(WAV_DIR):
    if not fname.endswith(".wav"):
        continue

    wav_path  = WAV_DIR  / fname
    rttm_path = RTTM_DIR / fname.replace(".wav", ".rttm")
    if not rttm_path.exists():
        print(f"[WARN] RTTM missing for {fname} – skipped")
        continue

    try:
        wav = preprocess_wav(str(wav_path))

        # embed_utterance → (utter_emb, partial_embs, partial_slices)
        _, partial_embs, _ = encoder.embed_utterance(wav, return_partials=True)
        emb_dim = partial_embs.shape[1]           # 256 or 512

        if len(partial_embs) < 2:                 # very short clip fallback
            partial_embs = np.vstack([partial_embs, partial_embs])

        # ---------- statistical pooling ----------
        mean_emb   = partial_embs.mean(0)
        std_emb    = partial_embs.std(0)
        median_emb = np.median(partial_embs, 0)
        iqr_emb    = np.percentile(partial_embs, 75, 0) - np.percentile(partial_embs, 25, 0)

        # ---------- K-elbow count estimate ----------
        Ks, inertias = [1, 2, 3, 4, 5], []
        for k in Ks:
            km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(partial_embs)
            inertias.append(km.inertia_)
        drops  = np.diff(inertias) / inertias[:-1]
        pred_k = 1 + np.argmax(drops > -0.05)        # more sensitive
        stat_row = {"k_elbow": float(pred_k)}

        # flatten stats
        stat_row.update({f"emb_mean_{i}":   mean_emb[i]   for i in range(emb_dim)})
        stat_row.update({f"emb_std_{i}":    std_emb[i]    for i in range(emb_dim)})
        stat_row.update({f"emb_median_{i}": median_emb[i] for i in range(emb_dim)})
        stat_row.update({f"emb_iqr_{i}":    iqr_emb[i]    for i in range(emb_dim)})

        # metadata
        label = count_unique_speakers(rttm_path)
        stat_row["filename"]     = fname
        stat_row["num_speakers"] = label
        rows_stats.append(stat_row)

        # ---------- save per-clip sequence ----------
        npz_path = SEQ_DIR / f"{fname}.npz"
        np.savez_compressed(npz_path, seq=partial_embs.astype(np.float32))
        rows_seq.append({"filename": fname,
                         "npz": str(npz_path),
                         "num_speakers": label})

        print(f"[INFO] processed {fname}")

    except Exception as e:
        print(f"[ERROR] {fname}: {e}")

# ----------------------------------------------------------------------
pd.DataFrame(rows_stats).to_csv(STATS_CSV, index=False)
pd.DataFrame(rows_seq).to_csv(SEQ_CSV,   index=False)

print("\nStats CSV  →", STATS_CSV)
print("\nSeq index →", SEQ_CSV)
print("\nNPZ files →", SEQ_DIR)