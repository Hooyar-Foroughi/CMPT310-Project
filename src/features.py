"""
Feature-extraction utilities used by every pipeline.

get_frame_embeddings(wav_path, rate=8) -> np.ndarray
pooled_vector(frame_embeds)            -> np.ndarray
load_true_count(rttm_path)             -> int
"""

from __future__ import annotations
import os
import numpy as np
from scipy.stats import skew, kurtosis
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.metrics.pairwise import cosine_distances

# Globals
encoder = VoiceEncoder("cpu")           # load once and share
EMB_DIM = 512                           # Resemblyzer default
__all__ = [
    "get_frame_embeddings",
    "pooled_vector",
    "load_true_count",
    "frame_distance_stats",
    "EMB_DIM",
]

# RTTM helper
def load_true_count(rttm_path):
    # Return the number of unique speaker IDs in an RTTM file
    speakers: set[str] = set()
    with open(rttm_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 8:
                speakers.add(parts[7])
    return len(speakers)

# Embedding extraction
def get_frame_embeddings(wav_path, rate=8):
    """
    Run Resemblyzer on a WAV file and return (N_frames, EMB_DIM) embeddings.
    Silence is trimmed by Resemblyzer's internal VAD.

    Parameters
    ----------
    wav_path : str
    rate     : int   Frames per second (default 8 to reduce RAM).

    Returns
    -------
    np.ndarray[float32]  shape = (N, 512)
    """
    wav = preprocess_wav(wav_path)
    _, embeds, _ = encoder.embed_utterance(
        wav, return_partials=True, rate=rate
    )
    return embeds.astype(np.float32, copy=False)

# Frame-distance helper
def frame_distance_stats(embeds):
    """
    Mean and standard deviation of pairwise cosine distances
    (cheap diversity proxy).

    Returns (mean_dist, std_dist)
    """
    if len(embeds) < 2:
        return 0.0, 0.0
    dists = cosine_distances(embeds)
    iu = np.triu_indices_from(dists, k=1)
    flat = dists[iu]
    return float(flat.mean()), float(flat.std())

# Pooling
def pooled_vector(embeds):
    """
    Produce a fixed‑length feature vector per clip made of three blocks:

    1. **Frame‑wise statistics** (per dimension, 512 dims each):
       mean, std, median, min, max, skew, kurtosis,
       10‑th, 25‑th, 50‑th, 75‑th, 90‑th percentiles,
       mean|std of first‑order delta (absolute diff between consecutive frames).

    2. **Global diversity**: mean & std of pairwise cosine distances (2 scalars)

    3. **Duration**        : number of frames (1 scalar)

    Total length = 512 × 12  + 2 + 1 = 6147.

    Returns
    -------
    np.ndarray[float32]  shape = (6147,)
    """
    # Core per‑dimension stats
    mu   = embeds.mean(axis=0)
    sd   = embeds.std(axis=0)
    med  = np.median(embeds, axis=0)
    mn   = embeds.min(axis=0)
    mx   = embeds.max(axis=0)
    sk   = skew(embeds, axis=0, bias=False)
    ku   = kurtosis(embeds, axis=0, bias=False)

    # Percentiles (5 × 512)
    p10, p25, p50, p75, p90 = np.percentile(
        embeds, [10, 25, 50, 75, 90], axis=0
    )

    # First‑order Δ embeddings
    if len(embeds) > 1:
        delta = np.abs(np.diff(embeds, axis=0))
        d_mu  = delta.mean(axis=0)
        d_sd  = delta.std(axis=0)
    else:  # single frame
        d_mu = np.zeros(embeds.shape[1], dtype=np.float32)
        d_sd = np.zeros_like(d_mu)

    # Pairwise diversity
    pd_mean, pd_std = frame_distance_stats(embeds)

    dur   = np.array([len(embeds)], dtype=np.float32)

    vec = np.hstack(
        [
            mu, sd, med, mn, mx, sk, ku,
            p10, p25, p50, p75, p90,
            d_mu, d_sd,
            [pd_mean, pd_std],
            dur,
        ]
    ).astype(np.float32)

    return vec