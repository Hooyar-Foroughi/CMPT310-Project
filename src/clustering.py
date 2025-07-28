"""
Speaker-count clustering utilities.

run_clustering(embeds, method="spectral")
    Returns the predicted number of speakers (k) in range 2-5.

Functions are stateless; caller passes the frame-level
embeddings ndarray of shape (N_frames, dim).
"""

from __future__ import annotations
import numpy as np
import sklearn.metrics as sm
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
import hdbscan

# Spectral clustering 
def spectral_best_k(emb, k_min, k_max):
    best_k, best_score = k_min, -1
    for k in range(k_min, k_max + 1):
        labels = SpectralClustering(
            n_clusters=k,
            affinity="rbf",            
            random_state=0,
        ).fit_predict(emb)
        try:
            s = sm.silhouette_score(emb, labels)  # euclidean metric
            if s > best_score:
                best_k, best_score = k, s
        except ValueError:
            pass
    return best_k

# Agglomerative clustering 
def agglomerative_best_k(emb, k_min, k_max):
    best_k, best_score = k_min, -1
    for k in range(k_min, k_max + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(emb)
        try:
            score = sm.silhouette_score(emb, labels, metric="cosine")
            if score > best_score:
                best_k, best_score = k, score
        except ValueError:
            pass
    return best_k

# Tuned HDBSCAN (auto-k) clustering
def hdbscan_count(emb):
    """
    Returns k in range 2-5 using HDBSCAN with cosine metric
    and size threshold to drop micro-clusters.
    """
    emb64 = emb.astype(np.float64, copy=False)
    clusterer = hdbscan.HDBSCAN(
        metric="cosine",
        algorithm="generic",     # brute-force for cosine
        min_cluster_size=8,
        min_samples=4,
    )
    labels = clusterer.fit_predict(emb64)
    valid = labels[labels != -1]
    if valid.size == 0:
        return 2
    counts = np.bincount(valid)
    size_thresh = max(3, int(0.05 * len(emb)))  # ≥ 5 % of frames
    k = (counts >= size_thresh).sum()
    return int(np.clip(k, 2, 5))

# Entry point
def run_clustering(embeds, method="spectral", k_min=2, k_max=5):
    """
    Predict speaker count.

    Parameters
    ----------
    embeds : ndarray  (N_frames, dim)
    method : 'spectral' | 'agglomerative' | 'hdbscan'
    k_min, k_max : int   search range for k (ignored by hdbscan)

    Returns
    -------
    int  in [k_min, k_max]   (or auto-k for hdbscan)
    """
    method = method.lower()
    if method == "spectral":
        return spectral_best_k(embeds, k_min, k_max)
    if method == "agglomerative":
        return agglomerative_best_k(embeds, k_min, k_max)
    if method == "hdbscan":
        return hdbscan_count(embeds)
    raise ValueError(f"Unknown clustering method '{method}'")