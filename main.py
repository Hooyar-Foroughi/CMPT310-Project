import os
from utils.audio_utils import load_audio, embed_audio
from utils.clustering import estimate_speakers

def main(audio_path):
    print(f"\n[INFO] Loading audio: {audio_path}")
    wav = load_audio(audio_path)

    print("\n[INFO] Extracting speaker embeddings...")
    embeddings = embed_audio(wav)

    print("\n[INFO] Estimating number of speakers...")
    num_speakers = estimate_speakers(embeddings)

    print(f"\nEstimated number of speakers: {num_speakers}")

if __name__ == "__main__":
    audio_path = "data/wav/test1.wav"
    main(audio_path)