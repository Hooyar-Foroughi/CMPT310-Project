from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import librosa
import os   

def extract_segments(audio_path, segment_duration=1.0):
    y, sr = librosa.load(audio_path, sr=None)
    segments = []
    for i in range(0, len(y), int(segment_duration * sr)):
        segment = y[i:i+int(segment_duration * sr)]
        segments.append(segment)
    return segments

def embed_audio(wav, encoder=None, rate=16):
    if encoder is None:
        encoder = VoiceEncoder()
    
    # Preprocess the audio for the encoder
    wav_processed = preprocess_wav(wav)
    
    _, cont_embeds, wav_splits = encoder.embed_utterance(
        wav_processed, 
        return_partials=True, 
        rate=rate
    )
    
    return cont_embeds, wav_splits

