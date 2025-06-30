from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import librosa

def load_audio(file_path, sample_rate=16000):
    wav = preprocess_wav(file_path)
    return wav

def embed_audio(wav, encoder=None, rate=16):
    if encoder is None:
        encoder = VoiceEncoder()
    _, cont_embeds, _ = encoder.embed_utterance(wav, return_partials=True, rate=rate)
    return cont_embeds