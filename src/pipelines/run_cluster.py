"""
Run unsupervised clustering (spectral | agglomerative | hdbscan) over a folder
of WAV clips and evaluate against RTTM ground truth when present.

Usage
-----
python -m src.pipelines.run_cluster \
       data/wav/            \
       --method spectral    \
       --rate 8             \
       --workers 4
"""

from __future__ import annotations
import argparse, os, glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

from src.features import get_frame_embeddings, load_true_count
from src.clustering import run_clustering
from src.evaluate import evaluate_dataframe

# Worker
def process_file(wav_path, rate, method, rttm_root, k_min, k_max):
    base = os.path.splitext(os.path.basename(wav_path))[0]
    try:
        emb = get_frame_embeddings(wav_path, rate=rate)
        k = run_clustering(emb, method=method, k_min=k_min, k_max=k_max)
    except Exception as exc:
        return {"filename": base, "pred": None, "error": str(exc)}

    rttm_path = os.path.join(rttm_root, f"{base}.rttm")
    true_cnt  = load_true_count(rttm_path) if os.path.exists(rttm_path) else None
    return {"filename": base, "pred": k, "true": true_cnt, "error": None}

# CLI
def cli():
    parser = argparse.ArgumentParser(description="Unsupervised speaker-count clustering")
    parser.add_argument("wav_dir", help="WAV file or folder containing WAV files")
    parser.add_argument(
        "--method",
        choices=["spectral", "agglomerative", "hdbscan"],
        default="spectral",
        help="Clustering backend",
    )
    parser.add_argument("--rate", type=int, default=8, help="Partial frame rate (Hz)")
    parser.add_argument("--rttm_dir", default="data/rttm", help="RTTM ground-truth folder")
    parser.add_argument("--workers", type=int, default=2, help="Thread pool size")
    parser.add_argument("--kmin", type=int, default=2, help="Minimum k to test")
    parser.add_argument("--kmax", type=int, default=5, help="Maximum k to test")
    parser.add_argument("--out_dir", default="results", help="Where to save CSV/TXT")
    args = parser.parse_args()

    # Accept either a single .wav file OR a directory of .wav files
    if os.path.isfile(args.wav_dir) and args.wav_dir.lower().endswith(".wav"):
        wav_files = [args.wav_dir]
    else:
        wav_files = glob.glob(os.path.join(args.wav_dir, "*.wav"))

    if not wav_files:
        raise SystemExit(f"No WAVs found in {args.wav_dir}")

    # parallel processing
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                process_file,
                wp,
                args.rate,
                args.method,
                args.rttm_dir,
                args.kmin,
                args.kmax,
            )
            for wp in wav_files
        ]
        for fut in tqdm(futures, desc="Processed", unit="file"):
            rows.append(fut.result())

    df = pd.DataFrame(rows)

    # --- demo mode: only one file ------------------------------------
    if len(df) == 1:
        rec = df.iloc[0]
        msg = f"{rec['filename']} → {rec['pred']} speaker(s)"
        if pd.notna(rec.get('true')):
            msg += f"  (true={int(rec['true'])})"
        print(msg)
        return

    if "error" in df.columns and df["error"].notna().any():
        errs = df[df["error"].notna()]
        print(f"[WARN] {len(errs)} files had errors — see below:")
        print(errs[["filename", "error"]].head(10))

    # evaluation + save 
    out_prefix = os.path.join(args.out_dir, f"cluster_{args.method}")
    report = evaluate_dataframe(df[["filename", "true", "pred"]], out_prefix=out_prefix)
    print("\nClassification Report:\n", report)
    print(f"[INFO] Results written to {out_prefix}.csv / .txt")

if __name__ == "__main__":
    cli()