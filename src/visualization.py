# ─────────────────────────────────────────────────────────────────────────────
#  src/visualization.py
#  quick plots for speaker-count experiments
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, os, glob, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import datetime

SAVE_DIR = "visualizations"  

def load_latest(results_dir="results"):
    csvs = sorted(glob.glob(os.path.join(results_dir, "*.csv")), key=os.path.getmtime)
    if not csvs:
        raise SystemExit(f"No CSVs found in '{results_dir}'. Run an experiment first.")
    return csvs[-1]

# prettier labels: 2 → “2 spk”, etc.
def pretty(lab):
    k_offset = 2 if lab <= 3 else 0   # empirical: 0‑based classes → 2‑5 speakers
    return f"{int(lab + k_offset)} spk"

def plot_report(df, title, save_png=True):
    df = df.dropna(subset=["true", "pred"])
    y_true, y_pred = df["true"].astype(int), df["pred"].astype(int)
    # ── align label coding with clustering output ────────────────
    # If labels are 0‑3 we assume they stand for 2‑5 speakers;
    # shift everything by +2 so the plot reflects real counts.
    if y_true.min() < 2 and y_true.max() <= 3:
        y_true = y_true + 2
        y_pred = y_pred + 2
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    labs_readable = [pretty(l) for l in labels]

    # ── metrics ────────────────────────────────────────────────────────────
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=None)

    fig = plt.figure(figsize=(10, 4))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # — confusion matrix —
    ax0 = fig.add_subplot(gs[0])
    im  = ax0.imshow(cm, cmap="Blues")
    ax0.set_title("Confusion matrix")
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("True")
    ax0.set_xticks(range(len(labels)), labs_readable, rotation=45, ha="right")
    ax0.set_yticks(range(len(labels)), labs_readable)
    # cell text
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax0.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() * .6 else "black")
    fig.colorbar(im, ax=ax0, fraction=.046)

    # — precision / recall / F1  —
    ax1 = fig.add_subplot(gs[1])
    x   = np.arange(len(labels))
    ax1.bar(x-.25, prec,  0.25, label="Precision")
    ax1.bar(x,      rec,  0.25, label="Recall")
    ax1.bar(x+.25,  f1,   0.25, label="F1")
    ax1.set_xticks(x, labs_readable, rotation=45, ha="right")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score")
    ax1.set_title("Per-class metrics")
    ax1.legend()

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if save_png:
        os.makedirs(SAVE_DIR, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(SAVE_DIR, f"{title}_{ts}.png")
        fig.savefig(path, dpi=180)
        print(f"[INFO] Figure saved → {path}")
    plt.show()

def plot_clusters(
    embeddings, 
    labels, 
    title="Cluster scatter",
    save_png=True,
    ax=None
):
    """
    embeddings : (N, dim)   frame-level embeddings
    labels     : (N,)       int labels (cluster id or true spk id)
    """
    X, labs = embeddings, labels

    reducer = PCA(n_components=2, random_state=0)
    XY = reducer.fit_transform(X)

    plt.figure(figsize=(6, 5))
    # --- build a colormap containing exactly n = #clusters colours ----------
    n_cls = int(labs.max()) + 1
    cmap  = plt.cm.get_cmap("tab10", n_cls)          # truncate tab10 -> n colours
    scatter = plt.scatter(
        XY[:, 0], XY[:, 1],
        c=labs,
        cmap=cmap,            # << use truncated cmap
        vmin=0, vmax=n_cls-1, # << ensure colour-bar spans 0 … n-1 only
        s=8, alpha=.7, edgecolors="none"
    )
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    cbar = plt.colorbar(scatter, shrink=.8)
    # show integer tick marks only (0,1,2,…)
    cbar.set_ticks(np.arange(int(labs.max()) + 1))
    cbar.set_ticklabels(np.arange(int(labs.max()) + 1))
    cbar.set_label("cluster / speaker id")
    plt.tight_layout()

    if save_png:
        os.makedirs(SAVE_DIR, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(SAVE_DIR, f"{title.replace(' ', '_')}_{ts}.png")
        plt.savefig(path, dpi=180)
        print(f"[INFO] Cluster figure saved → {path}")
    plt.show()

def main():
    ap = argparse.ArgumentParser("Visualise result CSV")
    ap.add_argument("csv", nargs="?", help="results/…csv  (defaults to newest)")
    args = ap.parse_args()

    csv_path = args.csv or load_latest()
    tag      = os.path.splitext(os.path.basename(csv_path))[0]
    df       = pd.read_csv(csv_path)
    plot_report(df, tag)

def main_cluster_scatter():
    """CLI: python -m src.visualization cluster_scatter <wav> [--method spectral]"""
    import argparse, soundfile as sf
    from src.features import get_frame_embeddings, load_true_count
    from src.clustering import run_clustering

    ap = argparse.ArgumentParser("Cluster scatter of a single WAV")
    ap.add_argument("wav")
    ap.add_argument("--method", default="spectral",
                    choices=["spectral", "agglomerative", "hdbscan"])
    ap.add_argument("--rate", type=int, default=8)
    ap.add_argument("--rttm_dir", default="data/rttm",
                    help="Folder containing *.rttm files (with same basename as WAV)")
    ap.add_argument("--k_range", default="2-5",
                    help="Generate scatter for every k in this inclusive range, e.g. '2-5'. "
                         "Use a single number (e.g. '3') to plot just that k.")
    args = ap.parse_args()

    emb = get_frame_embeddings(args.wav, rate=args.rate)
    # Try to load true speaker count
    base      = os.path.splitext(os.path.basename(args.wav))[0]
    rttm_path = os.path.join(args.rttm_dir, f"{base}.rttm")
    true_k    = load_true_count(rttm_path) if os.path.exists(rttm_path) else None

    # ------- run full clustering once to know the “chosen” k --------------
    chosen_k = run_clustering(
        emb,
        method=args.method,
        k_min=2,              # same search range the pipeline uses
        k_max=5,
    )
    print(f"[INFO] Pipeline would choose k = {chosen_k}")
    if true_k is not None:
        print(f"[INFO] Ground-truth k = {true_k}")

    # --- determine which k values we should visualise ------------------------
    if "-" in args.k_range:
        k_lo, k_hi = map(int, args.k_range.split("-"))
        k_values   = list(range(k_lo, k_hi + 1))
    else:
        k_values   = [int(args.k_range)]

    msg = f"Generating scatter for k={','.join(map(str,k_values))}"
    if true_k is not None:
        msg += f"  (true={true_k})"
    print(msg)

    from sklearn.cluster import SpectralClustering, AgglomerativeClustering
    for k in k_values:
        # recompute cluster labels with *fixed* k
        if args.method == "spectral":
            lab = SpectralClustering(
                n_clusters=k,
                affinity="rbf",
                random_state=0
            ).fit_predict(emb)
        elif args.method == "agglomerative":
            lab = AgglomerativeClustering(
                n_clusters=k,
                linkage="average",
                metric="cosine"
            ).fit_predict(emb)
        else:  # hdbscan (auto-k) – fall back to original behaviour
            import hdbscan
            lab = hdbscan.HDBSCAN(
                metric="cosine",
                algorithm="generic",
                min_cluster_size=8,
                min_samples=4,
            ).fit_predict(emb)
            k = len(set(lab[lab >= 0])) or 1  # recalc k for title

        title = (
            f"{os.path.basename(args.wav)}  "
            f"({args.method}, rate={args.rate}, k={k}"
        )
        if true_k is not None:
            title += f", true={true_k}"
        title += ")"
        plot_clusters(emb, lab, title, save_png=True)


if __name__ == "__main__":
    # decide which sub-mode to run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cluster_scatter":
        sys.argv.pop(1)          # remove the flag
        main_cluster_scatter()
    else:
        main()