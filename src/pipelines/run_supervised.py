"""
Train or use a supervised classifier on pooled features.

Usage
-----
# Train MLP (default)
python -m src.pipelines.run_supervised train --algo mlp --csv features_pooled.csv

# Train XGBoost
python -m src.pipelines.run_supervised train --algo xgb

# Predict on one clip with trained model
python -m src.pipelines.run_supervised predict demo.wav --model models/speaker_count_mlp.pkl

# Predict on a folder (writes CSV)
python -m src.pipelines.run_supervised predict data/wav/ --model models/speaker_count_mlp.pkl
"""

from __future__ import annotations
import argparse, os, glob, joblib, numpy as np, pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values*",
    category=UserWarning,
    module="sklearn.impute",
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None 

from src.features import get_frame_embeddings, pooled_vector

MODELS_DIR = "models"

def train_model(csv_path, algo="mlp"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    # Drop columns that are entirely NaN (prevents imputer warnings)
    df = df.dropna(axis=1, how="all")
    df = df[df["label"].between(2, 5)]
    # ---- drop labels that appear only once --------------------------
    counts = df["label"].value_counts()
    rare_labels = counts[counts < 2].index.tolist()
    if rare_labels:
        print(f"[WARN] Dropping tiny classes {rare_labels} (n<2).")
        df = df[~df["label"].isin(rare_labels)]
    # Re‑index labels so they are consecutive ints starting at 0
    original_labels = sorted(df["label"].unique())
    label2int = {lab: idx for idx, lab in enumerate(original_labels)}
    int2label = {v: k for k, v in label2int.items()}
    df["label_idx"] = df["label"].map(label2int)

    # ==== prepare features ====================================================
    feature_cols = [
        c for c in df.columns if c not in ("filename", "label", "label_idx")
    ]  # numeric feature names (= original indices as strings)
    # Persist positions of retained features
    try:
        # works when columns are "0", "1", … (legacy format)
        valid_idx = np.array([int(c) for c in feature_cols], dtype=np.int32)
    except ValueError:
        # new format uses names like "f0", "pd_mean", etc. – fall back to positions
        valid_idx = np.arange(len(feature_cols), dtype=np.int32)

    X = df[feature_cols].values
    y = df["label_idx"].values.astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fit_params = {}

    if algo == "mlp":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("vt", VarianceThreshold(threshold=1e-4)),
                ("pca", PCA(n_components=100, whiten=True, random_state=42)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        solver="adam",
                        max_iter=400,
                        random_state=42,
                    ),
                ),
            ]
        )

        # --- 5-fold CV *without* sample-weights --------------------------
        cv_scores = cross_val_score(
        pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1
        )
        print(
            "CV accuracy: {:.2f} ± {:.2f}%".format(
                cv_scores.mean() * 100, cv_scores.std() * 100
            )
        )
        y_pred_cv = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        # print(classification_report(y, y_pred_cv, digits=3, zero_division=0))
        # print("Confusion:\n", confusion_matrix(y, y_pred_cv))
        report = classification_report(
            [int2label[i] for i in y],
            [int2label[i] for i in y_pred_cv],
            digits=3,
            zero_division=0,
        )
        cm = confusion_matrix(
            [int2label[i] for i in y],
            [int2label[i] for i in y_pred_cv],
        )
        print(report)
        print("Confusion:\n", cm)
        print("-" * 60)

        # --- fit on full data *with* weights -----------------------------
        sample_w = compute_sample_weight("balanced", y)
        pipe.fit(X, y, clf__sample_weight=sample_w)
        # Persist the indices of the retained features
        pipe.valid_idx_ = valid_idx
    elif algo == "xgb":
        if XGBClassifier is None:
            raise SystemExit("xgboost not installed: pip install xgboost")

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("vt", VarianceThreshold(threshold=1e-4)),
                ("pca", PCA(n_components=100, whiten=True, random_state=42)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(np.unique(y)),
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        cv_scores = cross_val_score(
        pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1
        )
        print(
            "CV accuracy: {:.2f} ± {:.2f}%".format(
                cv_scores.mean() * 100, cv_scores.std() * 100
            )
        )
        y_pred_cv = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        # print(classification_report(y, y_pred_cv, digits=3, zero_division=0))
        # print("Confusion:\n", confusion_matrix(y, y_pred_cv))
        report = classification_report(
            [int2label[i] for i in y],
            [int2label[i] for i in y_pred_cv],
            digits=3,
            zero_division=0,
        )
        cm = confusion_matrix(
            [int2label[i] for i in y],
            [int2label[i] for i in y_pred_cv],
        )
        print(report)
        print("Confusion:\n", cm)
        print("-" * 60)

        sample_w = compute_sample_weight("balanced", y)
        pipe.fit(X, y, clf__sample_weight=sample_w)
        # Persist the indices of the retained features
        pipe.valid_idx_ = valid_idx
    elif algo == "hgb":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("vt", VarianceThreshold(threshold=1e-4)),
                ("pca", PCA(n_components=100, whiten=True, random_state=42)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_depth=3,
                        l2_regularization=0.1,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy",
                                    n_jobs=-1)
        print("CV accuracy: {:.2f} ± {:.2f}%".format(cv_scores.mean()*100,
                                                     cv_scores.std()*100))
        y_pred_cv = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        # print(classification_report(y, y_pred_cv, digits=3, zero_division=0))
        # print("Confusion:\n", confusion_matrix(y, y_pred_cv))
        report = classification_report(
            [int2label[i] for i in y],
            [int2label[i] for i in y_pred_cv],
            digits=3,
            zero_division=0,
        )
        cm = confusion_matrix(
            [int2label[i] for i in y],
            [int2label[i] for i in y_pred_cv],
        )
        print(report)
        print("Confusion:\n", cm)
        print("-" * 60)
        pipe.fit(X, y)
        # Persist the indices of the retained features
        pipe.valid_idx_ = valid_idx
    else:
        raise ValueError(f"Unknown algo '{algo}'")
    
    # ---- persist evaluation to results/ ---------------------------
    os.makedirs("results", exist_ok=True)
    txt_path = os.path.join("results", f"sup_{algo}.txt")
    with open(txt_path, "w") as fh:
        fh.write(
            "CV accuracy: {:.2f} ± {:.2f}%\n\n".format(
                cv_scores.mean() * 100, cv_scores.std() * 100
            )
        )
        fh.write(report)
        fh.write("\nConfusion:\n")
        fh.write(np.array2string(cm))
    # optional: save per‑sample predictions with filenames
    csv_path = os.path.join("results", f"sup_{algo}.csv")
    df_out = pd.DataFrame(
        {
            "filename": df["filename"].values,
            "pred": [int2label[i] for i in y_pred_cv],
            "true": [int2label[i] for i in y],
        }
    )
    df_out.to_csv(csv_path, index=False)
    print(f"[INFO] Results written to {csv_path} / .txt")

    print("[INFO] Model trained on full data.")

    # save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"speaker_count_{algo}.pkl")
    joblib.dump(pipe, model_path)
    print(f"[INFO] Saved → {model_path}")

# PREDICT
def predict_one(pipe, wav_path, rate=8):
    emb  = get_frame_embeddings(wav_path, rate=rate)
    vec  = pooled_vector(emb)
    if hasattr(pipe, "valid_idx_"):
        vec = vec[pipe.valid_idx_]
    feat = vec.reshape(1, -1)
    return int(pipe.predict(feat)[0])

def run_predict(model_path, target, rate=8, rttm_dir=None):
    pipe = joblib.load(model_path)
    # ----- single WAV ----------------------------------------------------
    if os.path.isfile(target) and target.lower().endswith(".wav"):
        k = predict_one(pipe, target, rate)

        # look for RTTM: first sibling, then the provided rttm_dir
        base = os.path.splitext(os.path.basename(target))[0]
        # 1) sibling
        rttm_path = os.path.splitext(target)[0] + ".rttm"
        # 2) dedicated directory
        if not os.path.exists(rttm_path) and rttm_dir:
            rttm_path = os.path.join(rttm_dir, base + ".rttm")
        if os.path.exists(rttm_path):
            from src.features import load_true_count
            true_k = load_true_count(rttm_path)
            print(f"{os.path.basename(target)} → {k} speakers  (true={true_k})")
        else:
            print(f"{os.path.basename(target)} → {k} speakers")

        return

    wav_files = glob.glob(os.path.join(target, "*.wav"))
    rows = []
    for wp in wav_files:
        k = predict_one(pipe, wp, rate)
        rows.append({"filename": os.path.basename(wp), "pred": k})

    df = pd.DataFrame(rows)
    # --- single‑file shortcut: just print and exit -------------------
    if len(rows) == 1 and rows[0].get("true") is None:
        single = rows[0]
        print(f"{single['filename']} → {single['pred']} speakers")
        return

    df.to_csv(os.path.splitext(model_path)[0] + "_predictions.csv", index=False)
    print(f"[INFO] Saved predictions → {os.path.splitext(model_path)[0]}_predictions.csv")

# CLI
def cli():
    p = argparse.ArgumentParser(description="Supervised speaker-count pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train model on features_pooled.csv")
    t.add_argument("--csv", default="features_pooled.csv")
    t.add_argument("--algo", choices=["mlp", "xgb", "hgb"], default="mlp")

    pr = sub.add_parser("predict", help="predict on wav or folder")
    pr.add_argument("path", help="wav file or folder")
    pr.add_argument(
        "--model",
        default=os.path.join(MODELS_DIR, "speaker_count_mlp.pkl"),
        help="Path to saved model *.pkl",
    )
    pr.add_argument("--rate", type=int, default=8)
    pr.add_argument("--rttm_dir", default=None, help="Directory containing RTTM files")

    args = p.parse_args()

    if args.cmd == "train":
        train_model(args.csv, algo=args.algo)
    elif args.cmd == "predict":
        run_predict(args.model, args.path, rate=args.rate, rttm_dir=args.rttm_dir)

if __name__ == "__main__":
    cli()