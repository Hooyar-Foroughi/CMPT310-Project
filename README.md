# CMPT310-Project

Install requirements: ***pip install -r requirements.txt***

Test/dev set audio files from: https://github.com/joonson/voxconverse?tab=readme-ov-file 

<h4> Currently we have 2 implementations to compare and build off of: </h4>

***Pooled-stats MLP:*** Converts each audio clip into a single summary vector using statistics (mean, std, etc.) of speaker embeddings. Then trains a small MLP classifier on those vectors.

```bash
python3 features/extractfeatures.py      # (re-run if audio clips changed)
python3 train-test/train.py
python3 train-test/evaluate.py
```

***Seq-embeddings Bi-LSTM:*** Keeps the full sequence of speaker embeddings per clip. Feeds them into a bidirectional LSTM with attention, which learns to weigh and combine time steps.

```bash
python3 features/extractfeatures.py      # generates .npz sequence files
python3 train-test/train_seq.py
python3 train-test/eval_seq.py
```

*Both methods share the same feature extractor (extractfeatures.py). Just pick the training pipeline you want to use afterward.*