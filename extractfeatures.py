import librosa
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__ == "__main__":

    filename='audio/hiyis.wav'  # Change this to extract from another file

    audioarray,sr=librosa.load(filename, sr=None)
    print("Audio array: ",audioarray)
    print("Sampling Rate: ",sr)

    # Key audio features for voice
    mfcc=librosa.feature.mfcc(y=audioarray,sr=sr)   # Small set of features (10-20) that describe the overall shape of a spectral envolope & models characteristics of human voice
    mfcc_delta = librosa.feature.delta(mfcc, order=1)   # Locate first and second derivative of mfcc
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    zero_crossing_rate=librosa.feature.zero_crossing_rate(y=audioarray)    # Where it crosses 0
    rms=librosa.feature.rms(y=audioarray)   # Compute root mean square for each frame
    spectral_centroid=librosa.feature.spectral_centroid(y=audioarray,sr=sr) # Locates the weighted mean/center mass of sound
    spectral_rolloff=librosa.feature.spectral_rolloff(y=audioarray,sr=sr) # Measure of shape of signal
    spectral_bandwidth=librosa.feature.spectral_bandwidth(y=audioarray,sr=sr)   # Measures the spread of the frequencies    
    f0, voiced_flag, voiced_probs = librosa.pyin(y=audioarray, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))    # Pitch and voice recognition **MAIN REASON THIS TAKES A LONG TIME**


    # # print("Mel Frequency Cepstral Coefficients: ",mfcc)
    # print("Mel Frequency Cepstral Coefficients: ",mfcc.shape)
    # # print("MFCC delta order 1: ",mfcc_delta)
    # print("MFCC delta order 1: ",mfcc_delta.shape)
    # # print("MFCC delta order 1: ",mfcc_delta2)
    # print("MFCC delta order 2: ",mfcc_delta2.shape)
    # print("zero_crossings: ",zero_crossing_rate)
    # print("zero_crossings: ",zero_crossing_rate.shape) 
    # # print("Root Mean Squared: ",rms)
    # print("Root Mean Squared: ",rms.shape)
    # # print("Pitch time series: ", f0)
    # print("Pitch time series: ", f0.shape)
    # # print("Pitch Voiced Flag: ",voiced_flag)
    # print("Pitch Voiced Flag: ",voiced_flag.shape)
    # # print("Pitch Voiced Probability: ",voiced_probs)
    # print("Pitch Voiced Probability: ",voiced_probs.shape)
    # # print("Spectral Centroid: ",spectral_centroid)
    # print("Spectral Centroid: ",spectral_centroid.shape)
    # # print("Spectral Rolloff: ",spectral_rolloff)
    # print("Spectral Rolloff: ",spectral_rolloff.shape)
    # # print("Spectral Bandwidth: ",spectral_bandwidth)
    # print("Spectral Bandwidth: ",spectral_bandwidth.shape)


num_mfcc_coeffs = mfcc.shape[0]

df = pd.DataFrame()

mfccs=np.split(mfcc,num_mfcc_coeffs)
mfcc_deltas=np.split(mfcc_delta,num_mfcc_coeffs)
mfcc_delta2s=np.split(mfcc_delta2,num_mfcc_coeffs)
for i in range(num_mfcc_coeffs):
    df[f'mfcc_{i}']=mfccs[i][0]
    df[f'mfcc_delta1_{i}']=mfcc_deltas[i][0]
    df[f'mfcc_delta2_{i}']=mfcc_delta2s[i][0]
df = df.reindex(columns=sorted(df.columns))
df['zero_crossing_rate']=zero_crossing_rate[0]
df['rms']=rms[0]
df['spectral_centroid']=spectral_centroid[0]
df['spectral_rolloff']=spectral_rolloff[0]
df['spectral_bandwidth']=spectral_bandwidth[0]
df['f0']=f0
df['voiced_flag']=voiced_flag
df['voiced_probs']=voiced_probs


print("\n--- DataFrame Head ---")
print(df.head())
print(f"\n--- DataFrame Shape ---")
print(f"Rows (frames): {df.shape[0]}")
print(f"Columns (features): {df.shape[1]}")


output_filepath = os.path.join("features", f"{os.path.splitext(os.path.basename(filename))[0]}.csv")

# Save the DataFrame to a CSV file
df.to_csv(output_filepath, index=False)