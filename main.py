from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# Load the audio file
# Replace 'audio/jyirt.wav' with the path to your multi-speaker audio file
wav_fpath = Path('audio/jyirt.wav') 
print("File name:", wav_fpath)
wav = preprocess_wav(wav_fpath)
# Initialize the VoiceEncoder
# You can specify "cuda" for GPU if available, otherwise it defaults to "cpu"
encoder = VoiceEncoder("cpu") 

# To get a single summary embedding for the entire utterance (less useful for diarization)
# embed = encoder.embed_utterance(wav)

# To get continuous embeddings for diarization, you need to set return_partials=True
# and a rate (how often an embedding is generated). A higher rate means more embeddings
# and potentially finer-grained diarization, but also more computation.
# A common rate is 16, meaning an embedding every 0.0625 seconds.
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

# print(f"Shape of continuous embeddings: {cont_embeds.shape}")
# print(f"Number of audio splits (segments): {len(wav_splits)}")

pplCount=0
highscore=0
model=None
print("\nNumber of clusters starting from 2-10")
for i in range(2,11):
    cluster_model = AgglomerativeClustering(n_clusters=i)
    # cluster_model=KMeans(n_clusters=i, n_init='auto')
    labels = cluster_model.fit_predict(cont_embeds)
    score=silhouette_score(cont_embeds,labels)
    print(score)
    if highscore < score:
        highscore=score
        pplCount=i
        model=cluster_model
print("\n")
print("People Count: ", pplCount)
print("Labels: ", model.fit_predict(cont_embeds))