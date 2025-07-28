from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

# Load the audio file
# Replace 'audio/jyirt.wav' with the path to your multi-speaker audio file
wav_fpath = Path('audio/jyirt.wav') 
print("File name:", wav_fpath)
wav = preprocess_wav(wav_fpath)
encoder = VoiceEncoder() 

# To get a single summary embedding for the entire utterance (less useful for diarization)
# embed = encoder.embed_utterance(wav)

# To get continuous embeddings for diarization, you need to set return_partials=True
# and a rate (how often an embedding is generated). A higher rate means more embeddings
# and potentially finer-grained diarization, but also more computation.
# A common rate is 16, meaning an embedding every 0.0625 seconds.
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

pplCount=0
highscore=0
model=None
modellabels=None
print("\nNumber of clusters starting from 2-10")
for i in range(2,11):
    # Agglomerative works best for standard clear cases (Faster but less sensitive)
    cluster_model = AgglomerativeClustering(n_clusters=i)

    # Spectural works best with noisy unclear cases (Slower but it's more sensitive)
    # cluster_model = SpectralClustering(n_clusters=i)

    # KMeans does not work because spherical radius does not properly capture each distinct voice
    # cluster_model=KMeans(n_clusters=i, n_init='auto')

    labels = cluster_model.fit_predict(cont_embeds)
    score=silhouette_score(cont_embeds,labels)
    print(score)
    if highscore < score:
        highscore=score
        pplCount=i
        model=cluster_model
        modellabels=labels
print("\n")
print("People Count: ", pplCount)
print("Labels: ", modellabels)

sample_rate = 16000 # Resemblyzer's default preprocessing sample rate of 16k

speaker_segments=[]
for i in range(pplCount):
    speaker_segments.append([])

for i, label in enumerate(modellabels):
    start_time_samples = wav_splits[i].start
    end_time_samples = wav_splits[i].stop
    
    # Convert from samples to seconds
    start_time_seconds = start_time_samples / sample_rate
    end_time_seconds = end_time_samples / sample_rate
    
    speaker_segments[label].append((start_time_seconds, end_time_seconds))

print("\nSpeaker Timestamps (in seconds):")
for i, segments in enumerate(speaker_segments):
    print(f"Speaker {i}:")
    for start, end in segments:
        print(f"  - [{start:.2f}s - {end:.2f}s]")

# You would then apply the consolidation logic as discussed before,
# using these now-correctly-calculated start_time_seconds and end_time_seconds.
# For a more coherent view, merge adjacent segments for the same speaker

coherent_speaker_segments=[]
for i in range(pplCount):
    coherent_speaker_segments.append([])

for speaker, segments in enumerate(speaker_segments):
    # Sort segments by start time
    segments.sort()

    current_start, current_end = segments[0]
    for i in range(1, len(segments)):
        next_start, next_end = segments[i]
        
        # If the next segment is immediately after or overlaps slightly, merge them
        if next_start - current_end < 0.1:  # 0.1s tolerance for merging
            current_end = next_end
        else:
            coherent_speaker_segments[speaker].append((current_start, current_end))
            current_start, current_end = next_start, next_end
    coherent_speaker_segments[speaker].append((current_start, current_end))

print("\nCoherent Speaker Timestamps:")
for i, segments in enumerate(coherent_speaker_segments):
    print(f"Speaker {i}:")
    for start, end in segments:
        print(f"  - [{start:.2f}s - {end:.2f}s]")