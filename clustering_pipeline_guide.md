CMPT 310 D200 - Summer 2025 - Group #35 - Hooyar Foroughizadeh, Victor Nguyen, Aaryan Wakchaure

## How to Perform Unsupervised Speaker Diarization with Spectral Clustering and Voice Embeddings
&nbsp; 
## 1. Introduction

Speaker diarization is the task of automatically segmenting an audio stream by identifying which speaker is speaking when. In this guide, you’ll learn how to experiment with our unsupervised diarization pipeline, which uses spectral clustering on voice embeddings to discover speaker groups without any labeled training data. This approach is especially powerful for real-world use cases where annotations are unavailable, making it easy to apply across diverse datasets.

You can explore the full project repository, including source code and detailed documentation, [here](https://github.com/Hooyar-Foroughi/CMPT310-Project).

&nbsp; 
## 2. Prerequisites

### Software Requirements

- Python 3.11
- Install project dependencies:

```bash
pip install -r requirements.txt
```

### Data Preparation

To run the clustering process described in this guide, you first need suitable audio files. Any `.wav` audio file will work, but for optimal performance, choose audio samples that:

- Are relatively short (preferably under 5 minutes in length)
- Contain clear speech segments with multiple distinct speakers

Using short and clearly segmented audio helps the clustering algorithm identify speakers accurately and quickly.

Optionally, if you want to evaluate the accuracy of your clustering predictions, you can also obtain corresponding ground-truth `.rttm` files. These RTTM files indicate the true number of speakers and their speaking intervals in each audio clip, making them very useful for assessing your clustering performance.

A convenient source for pre-labeled data is the VoxConverse dataset:

- Download test/dev `.wav` and `.rttm` files from the [VoxConverse](https://github.com/joonson/voxconverse) dataset.
- Place `.wav` audio files into the `data/wav/` folder.
- (Optional) Place the corresponding `.rttm` label files into the `data/rttm/` folder.

However, you're free to use your own audio recordings, even if you don't have ground-truth labels available. Simply ensure your `.wav` files are organized in the `data/wav/` directory of your project structure.

&nbsp; 
## 3. Step-by-Step: Running the Unsupervised Diarization Pipeline
&nbsp; 
### Step 1: Set up the Configuration File

Open your `config.yaml` file and set these parameters under the "General settings and paths" and "Unsupervised (cluster task)" sections:

- `task`: Set this to `cluster`

  ```yaml
  task: cluster
  ```

- `cluster_target`:

  Specify the path to the WAV file or folder. You can run the pipeline on either:
   - A **single .wav file**: to get a one-off prediction with no evaluation.
   - **A folder of .wav files**: Use this to run clustering on multiple files **and evaluate** the predictions against ground-truth labels. **Important:** If you use a folder for cluster_target, you must also have matching .rttm files (with the same base filenames) in the folder specified by rttm_dir (e.g., `rttm_dir: data/rttm`) so the system can evaluate accuracy.

  ```yaml
  cluster_target: data/wav/your_audio.wav  # single file
  # OR
  cluster_target: data/wav                 # entire folder
  ```


The following settings affect performance and accuracy significantly:

- `rate` *(Hz)*:

  The sampling rate for voice embeddings extraction. Higher values increase accuracy but require more memory and computational resources.

  Recommended starting point:

  ```yaml
  rate: 4
  ```

- `kmin` and `kmax` *(speaker range)*:

  Define the range of speaker counts to evaluate. The algorithm tests all counts within this range.

  *Note: the minimum number of speakers (kmin) must be atleast 2*
  
  Example configuration:

  ```yaml
  kmin: 2     # minimum speakers
  kmax: 5     # maximum speakers
  ```

  Keep the range realistic to manage computational load.

- `method` *(clustering algorithm)*:\
  Choose from:

  - `spectral`
  - `agglomerative`
  - `hdbscan`

  Recommended default for this guide:

  ```yaml
  method: spectral
  ```

- `workers` *(thread-pool size)*:\
  Number of CPU cores for parallel processing. Increasing this speeds up processing but can cause instability on weaker machines.

  Recommended default:

  ```yaml
  workers: 1
  ```

---
&nbsp; 
### Step 2: Run the Pipeline

Run your pipeline from the terminal at the project root:

```bash
python3 run.py
```

#### Under the Hood

The pipeline performs these steps automatically:

1. **Voice Embedding Extraction**: Transforms audio data into speaker-specific embedding vectors.
2. **Affinity Matrix Construction**: Measures pairwise similarities among embeddings.
3. **Spectral Decomposition**: Extracts eigenvectors representing embedding structure.
4. **K-means Clustering**: Partitions embeddings into `k` clusters for each value from `kmin` to `kmax`.
5. **Optimal Cluster Selection**: Chooses the best `k` based on criteria like eigen-gap or silhouette score.

--- 
### Troubleshooting Tips:

- **High memory usage or slow processing**:

  - **Cause**: `rate` too high.
  - **Fix**: Lower `rate` (e.g., from 16 to 4).

- **Very low accuracy**:

  - **Cause**: Incorrect `kmin` and `kmax`.
  - **Fix**: Adjust speaker count range realistically.



---
&nbsp; 
### Step 3: Inspect Results

If you clustered a **single file**, your terminal will only output the predicted number of speakers.

If you clustered a **folder**, the script will:
- Save a `.csv` file with predictions and ground truth values in your `results/` directory.
- Compare against RTTM files and save a `.txt` with metrics (e.g., accuracy, recall) in your `results/` directory.

CSV file (`results/cluster_spectral.csv`):

  ```csv
  filename,predicted,true,correct
  audio_sample.wav,3,3,True
  ```

TXT file (`results/cluster_spectral.txt`):

  ```
  Classification Report:
                    precision    recall  f1-score   support

            2           0.70       0.77       0.74       13
            3           0.50       0.45       0.47       11
            4           0.60       0.50       0.55       10

  Confusion Matrix:
  [[10  2  1]
   [ 3  5  3]
   [ 2  3  5]]
  ```

---
&nbsp; 
### Step 4: Visualizing Results

To visualize clustering, update `config.yaml`:

```yaml
task: cluster_scatter
scatter_wav: data/wav/your_audio.wav
scatter_method: spectral
scatter_rate: 4
```

To better understand how well the unsupervised clustering separates speakers in a given audio file, the cluster_scatter task visualizes the embedding space and the speaker clusters discovered by the algorithm.

This visualization task is controlled by the following **config options**:
- `scatter_method`: Specifies which clustering method to visualize. Options include spectral, agglomerative, or hdbscan. The choice of method will affect how clusters are formed and displayed. For this guide, we will use spectral clustering. 
- `scatter_rate`: Controls the temporal resolution of the extracted voice embeddings (in Hz). A higher rate (e.g., 16) means more embeddings per second, which gives finer resolution but may slow down processing and clutter the scatter plot.

When you run the cluster_scatter task on a .wav file, the system automatically performs clustering for each value of k 2-5. It then displays a separate scatter plot for each k, showing how the embeddings are grouped into speaker clusters.

This is useful for:
- Comparing the shape and separation of clusters at different values of k
- Understanding how clearly speakers are separated in the embedding space
- Visually identifying which k appears to give the most distinct clusters

These visualizations are saved in the visualizations/ directory. Each file includes the predicted speaker count, true label (if available), clustering method, and rate in the title for easy reference.

Once your `config.yaml` is setup for visualizing results, run:

```bash
python3 run.py
```

&nbsp; 
Example Visualizations:

| ![](https://i.imgur.com/94wVndU.png) | ![](https://i.imgur.com/yy0HikG.png) |
|:--:|:--:|
| Example 1 | Example 2 |

**Example 1:**
- In this scatter plot of gylzn.wav, we observe two distinct and well-separated clusters. This indicates a clean diarization result, where the algorithm correctly identified the presence of two speakers. The prediction aligns with the true label (true=2), confirming the effectiveness of clustering in this case.

**Example 2:**
- In leneg.wav, we see three visually identifiable clusters, and the algorithm predicts k=3. However, the ground truth is true=2. This discrepancy is due to the nature of the audio: a talk show clip featuring two speakers in conversation and a third source, the crowd, frequently clapping, which contributes acoustically and is picked up as a separate cluster. This example highlights how clustering can reveal distinct audio patterns, even when they aren’t part of the main speaker count.


&nbsp; 
## 4. Reflection and Next Steps

This unsupervised approach demonstrates a practical and effective method for separating speakers in audio clips without the need for labeled training data. By leveraging voice embeddings and clustering, we are able to uncover the natural structure present in the acoustic features of the audio. Spectral clustering is particularly powerful in this setting because it excels at capturing non-linear relationships between voice segments, enabling it to group similar speaker embeddings even in complex scenarios.

One of the unique strengths of our pipeline is its adaptive experimentation with cluster sizes. By evaluating a range of k values and visualizing the results, we gain deeper insight into which clustering scenario best represents the true speaker distribution. This provides a clear understanding of the model’s behavior in ambiguous or noisy situations.

Looking forward, the system could be enhanced by introducing a binary classifier that detects whether a given cluster represents a human voice or background/non-speech sounds (e.g., clapping, music, static). This would help filter out non-relevant audio components and refine the diarization output.

Overall, this project offers a flexible and extensible foundation for speaker diarization, providing balance between simplicity, interpretability, and effectiveness.


&nbsp;
&nbsp;
&nbsp; 
&nbsp;
##### License and Permission

We,  Group #35 of CMPT 310 - D200 Summer 2025, grant the instructor and Simon Fraser University permission to share and use this project, guide, and documentation for academic use.

