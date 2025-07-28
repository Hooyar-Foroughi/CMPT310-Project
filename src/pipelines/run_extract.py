"""
Build pooled-feature vectors for every WAV file in a folder.

Each row of the output CSV contains:
    filename, label, f0, f1, ..., fN         (label = #speakers from RTTM)

Usage (CLI)
-----------
python -m src.pipelines.run_extract data/wav/ \
       --rate 8 \
       --csv features_pooled.csv \
       --rttm_dir data/rttm \
       --workers 4
"""

from __future__ import annotations
import argparse, os, glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.features import (
    get_frame_embeddings,
    pooled_vector,
    load_true_count,
    EMB_DIM,
)

# Worker
def process_one(wav_path, rate, rttm_dir):
    base = os.path.splitext(os.path.basename(wav_path))[0]
    try:
        emb = get_frame_embeddings(wav_path, rate=rate)
        vec = pooled_vector(emb)                      # 3075-d by default
    except Exception as exc:
        print(f"[ERR] {base}: {exc}")
        return {}

    rttm_path = os.path.join(rttm_dir, f"{base}.rttm")
    label = load_true_count(rttm_path) if os.path.exists(rttm_path) else None
    row = {"filename": base, "label": label}
    row.update({f"f{i}": v for i, v in enumerate(vec)})
    return row

# CLI
def cli():
    ap = argparse.ArgumentParser(description="Extract pooled Resemblyzer features")
    ap.add_argument("wav_dir", help="Folder containing .wav clips")
    ap.add_argument("--rate", type=int, default=8, help="Partial frame rate (Hz)")
    ap.add_argument("--csv", default="features_pooled.csv", help="Output CSV path")
    ap.add_argument("--rttm_dir", default="data/rttm", help="Ground-truth RTTM folder")
    ap.add_argument("--workers", type=int, default=4, help="Thread pool size")
    args = ap.parse_args()

    wav_files = glob.glob(os.path.join(args.wav_dir, "*.wav"))
    if not wav_files:
        raise SystemExit(f"No .wav files found in {args.wav_dir}")

    header_written = False
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
         open(args.csv, "w", newline="") as fh:

        for res in tqdm(
            ex.map(process_one, wav_files,
                   [args.rate]*len(wav_files),
                   [args.rttm_dir]*len(wav_files)),
            total=len(wav_files),
            desc="Extracting", unit="file"):

            if not res:
                continue  # error already printed
            # Write header once
            if not header_written:
                fh.write(",".join(res.keys()) + "\n")
                header_written = True
            # Write row
            fh.write(",".join(map(str, res.values())) + "\n")

    print(f"[INFO] Saved pooled features → {args.csv}")

if __name__ == "__main__":
    cli()