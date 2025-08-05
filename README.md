# CMPT310 Speaker Diarization Project

This project implements both supervised and unsupervised methods for **speaker diarization**, the process of segmenting an audio stream by speaker identity. We experiment with a variety of techniques and evaluate their performance both quantitatively and visually to gain deeper insights into how diarization can be approached effectively. By comparing different clustering strategies, classifier pipelines, and preprocessing techniques, we aim to understand the impact of various design decisions.

A particularly promising approach comes from our unsupervised spectral clustering implementation. This method builds an affinity matrix from speaker embeddings and applies spectral decomposition followed by k-means clustering to separate speakers. What makes our implementation unique is that we explore a range of cluster values (k), evaluating each and selecting the best fit based on objective criteria like eigen-gap analysis. This enables the model to adapt dynamically to the actual number of speakers in a recording. The strength of this technique lies in its ability to uncover the natural structure of the data without requiring labeled examples. By combining voice embeddings, affinity refinement, and eigen-space projection, our approach can distinguish speakers based on subtle acoustic features.

To support analysis, we generate visualizations of the clustering output using scatter plots and histograms. These help us interpret how well the embeddings and clustering methods isolate speakers. Visual outputs include per-audio scatter plots of clustered embeddings and bar charts comparing true vs predicted number of speakers, allowing us to evaluate not just accuracy but clustering behavior.

## Features

- **Voice encoder-based feature extraction** from audio
- **Unsupervised clustering** using Spectral, Agglomerative, and HDBSCAN
- **Supervised classification** using Scikit-learn pipelines
- **Evaluation** using RTTM ground-truth files
- **Visualizations** of clustering and performance metrics
- **Timestamping** using Spectural and Agglomerative clustering
- **Real-time VAD with Playback** using Timestamp labels and Pygame to show speakers with coloured squares
---

## Directory Structure

```
├── data/
│   ├── wav/              # Raw .wav audio files
│   ├── rttm/             # Corresponding RTTM label files
│   ├── agtimestamp/      # Timestamps .csv created by Agglomerative Clustering
│   └── sptimestamp/      # Timestamp .csv created by Spectural Clustering
├── models/               # Saved supervised models
├── results/              # CSV results for predictions and clustering
├── visualizations/       # Clustering scatter plots
├── src/                  # Main source code
│   ├── pipelines/        # Supervised + Unsupervised logic
│   ├── features.py       # Feature extraction methods
│   ├── visualization.py  # Plotting tools
│   ├── filter_dataset.py # Filters our /data to keep appropriate samples
│   ├── clustering.py     # Implements unsupervised speaker count prediction
│   ├── evaluate.py       # Provides utilities to compute and save classification metrics
│   ├── timestampall.py   # Creates timestamp .csv files for every .wav audio file in wav/
│   ├── gettimestamp.py   # Manual program to create timestamp .csv files selectively, timestampall.py uses this
│   ├── audioplayer.py    # Displays one clustering model's timestamp .csv labels in real-time with audio that undergone VAD filtering in sync
│   └── playbothtimestamps.py # Displays both clustering model's timestamp .csv labels in real-time their own players at the same time
├── config.yaml           # Main configuration file
├── run.py                # CLI runner
└── requirements.txt      # Python dependencies
```

---

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the test and dev set `.wav` and `.rttm` files:
   - Source: [VoxConverse](https://github.com/joonson/voxconverse)
   - Place `.wav` files in `data/wav/`
   - Place `.rttm` files in `data/rttm/`

---

## Usage

All tasks are run via:

```bash
python run.py
```

The task is configured through `config.yaml`.

---

## Config (`config.yaml`) 

```yaml
...
```
---

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

---

## Acknowledgements

- [VoxConverse Dataset](https://github.com/joonson/voxconverse)
- [Resemblyzer Voice Encoder](https://github.com/resemble-ai/Resemblyzer)
- Built for SFU CMPT 310