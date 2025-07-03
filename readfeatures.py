import librosa
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__ == "__main__":      # Reads every file inside features
    for file in os.listdir("features"):
        path = os.path.join("features",file)
        print(pd.read_csv(path))