# CMPT310-Project

Install requirements: ***pip install -r requirements.txt***

Test/dev set audio files from: https://github.com/joonson/voxconverse?tab=readme-ov-file 

audio: For storing .wav files
agtimestamp: for generated csv timestamps with agglomerative model
sptimestamp: for generated csv timestamps with spatical model

run ```python timestampall.py``` to create timestamp csv files for all newly added .wav files in the audio directory.
run ```python gettimestamp.py {path/to/.wavfile} {"ag","sp"}``` to specifically create a timestamp csv for 1 .wav file with Agglomerative or Spatical model

Then you can run ```python audioplayer.py {path/to/.csv}``` to play the audio and display the speakers
You can also run ```python playbothtimestamps.py {path/to/.csv}``` to play both audio