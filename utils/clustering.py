from sklearn.cluster import KMeans
import numpy as np

def estimate_speakers(embeddings, max_speakers=6):
    scores = []
    for k in range(1, max_speakers+1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(embeddings)
        scores.append(kmeans.inertia_)  # Sum of squared distances

    diffs = np.diff(scores)
    elbow = np.argmin(diffs > -0.05 * diffs[0]) + 1  # crude elbow method
    return elbow + 1