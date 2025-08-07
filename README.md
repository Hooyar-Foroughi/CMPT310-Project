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

## Config

All experiment settings are defined in [`config.yaml`](config.yaml). To run any task, edit the file to specify the `task:` name and adjust related settings as needed. Below is an overview of the available tasks and how to configure each section:

#### Task Selector
Set the main task to run:
```yaml
task: cluster  # Options: cluster | extract | train | predict | visualize | cluster_scatter | timestampall
```

#### General Settings
These settings apply across most tasks:
```yaml
rate: 4             # Frame rate for embedding extraction (higher = more granularity, slower)
workers: 1          # Number of parallel threads to use
wav_dir: data/wav   # Folder containing input WAV files
rttm_dir: data/rttm # Folder of ground-truth RTTM files (used for evaluation)
out_dir: results    # Folder to save all .txt and .csv output
```

#### Unsupervised Clustering (`task: cluster`)
```yaml
cluster_target: data/wav/leneg.wav  # Set a single .wav file for prediction, or use 'data/wav/' to evaluate over all files
method: spectral                    # Clustering method: spectral | agglomerative | hdbscan
kmin: 2                             # Minimum number of clusters (used for evaluation loop)
kmax: 5                             # Maximum number of clusters
```
- If running on a **single file**, the script will output the predicted number of speakers.
- If running on a **folder of .wav files**, and RTTM files are present in `rttm_dir`, it will evaluate results and generate `.csv`/`.txt` output in `out_dir`.

#### Feature Extraction (`task: extract`)
```yaml
features_csv: features_pooled.csv  # Output path for extracted features
```
This extracts pooled features (e.g., MFCCs, RMS, pitch, etc.) from each file and stores them in a single `.csv`.

#### Supervised Training (`task: train`)
```yaml
algo: xgb  # Options: xgb | hgb | mlp
```
- Trains a model to predict number of speakers based on features.
- Uses `features_csv` from the `extract` task.

#### Supervised Prediction (`task: predict`)
```yaml
model_path: models/speaker_count_mlp.pkl  # Path to trained model
predict_target: data/wav/gylzn.wav        # File or folder to predict
```
- Predicts speaker count for each `.wav` file using the selected model.

#### Visualizing Results (`task: visualize`)
```yaml
viz_csv: results/sup_xgb.csv  # CSV file containing prediction and label results
```
- Produces confusion matrices and per-class precision/recall/F1 charts.

#### Scatter Plot Visualization (`task: cluster_scatter`)
```yaml
scatter_wav: data/wav/gylzn.wav     # Target .wav file to visualize
scatter_method: spectral            # Clustering method: spectral | agglomerative | hdbscan
scatter_rate: 16                    # Embedding frame rate (same meaning as general rate)
```
- Produces scatter plots for each `k` between `kmin` and `kmax` using clustering outputs.
- Displays how embeddings were grouped and whether the prediction matches the true number of speakers (if available).

#### Audio Playback (manual)
The `playaudio` and `playboth` tasks must be set directly via command-line and require a path to a `.csv` file:
```bash
python3 run.py playaudio results/my_results.csv
```
These are useful for listening to segments to validate clusters.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

---

## Acknowledgements

- [VoxConverse Dataset](https://github.com/joonson/voxconverse)
- [Resemblyzer Voice Encoder](https://github.com/resemble-ai/Resemblyzer)
- Built for SFU CMPT 310