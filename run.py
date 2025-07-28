#!/usr/bin/env python3
"""
run.py  –  Single entry-point launcher that reads config.yaml and performs
unsupervised clustering *or* supervised training / prediction with a simple
command.

Usage (from repo root)
---------------------
# Unsupervised clustering (default task)
python3 run.py               # uses config.yaml

# Explicit task selection
python run.py cluster        # same as above
python run.py train          # train model defined in config
python run.py predict demo.wav  --model models/speaker_count_mlp.pkl
"""

import argparse, yaml, os, sys

# Extend sys.path so we can "import src.*" when run from repo root
top_dir = os.path.dirname(os.path.abspath(__file__))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.pipelines import run_cluster, run_supervised, run_extract 

CONFIG_FILE = "config.yaml"

def load_cfg(path=CONFIG_FILE):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

def main():
    cfg = load_cfg()

    parser = argparse.ArgumentParser("CMPT310 unified launcher")
    sub   = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("cluster")
    sub.add_parser("extract")

    sub.add_parser("visualize")
  
    tr = sub.add_parser("train")
    tr.add_argument("--algo", choices=["mlp", "xgb"], default=cfg.get("algo", "mlp"))

    pr = sub.add_parser("predict")
    pr.add_argument("path", help="wav file or folder")
    pr.add_argument("--model", default="models/speaker_count_mlp.pkl")
    pr.add_argument("--rttm_dir", default=None, help="Directory containing RTTM files")

    args = parser.parse_args()
    task = args.cmd or cfg.get("task", "cluster")

    if task == "extract":
        print("[RUN] Building pooled features …")
        sys.argv = ["run_extract.py", cfg["wav_dir"], "--rate", str(cfg["rate"])]
        run_extract.cli()

    elif task == "cluster":
        print("[RUN] Unsupervised clustering …")
        cluster_path = cfg.get("cluster_target", cfg["wav_dir"])  # can be dir or a single .wav
        sys.argv = [
            "run_cluster.py",
            cluster_path,
            "--method",  cfg["method"],
            "--rate",    str(cfg["rate"]),
            "--workers", str(cfg["workers"]),
            "--kmin",    str(cfg.get("kmin", 2)),
            "--kmax",    str(cfg.get("kmax", 5)),
            "--rttm_dir", cfg["rttm_dir"],
            "--out_dir",  cfg["out_dir"],
        ]
        run_cluster.cli()

    elif task == "train":
        print("[RUN] Supervised training …")
        algo = getattr(args, "algo", cfg.get("algo", "mlp"))

        sys.argv = [
            "run_supervised.py",
            "train",
            "--algo", algo,
            "--csv", cfg.get("features_csv", "features_pooled.csv"),
        ]
        run_supervised.cli()

    elif task == "predict":
        print("[RUN] Supervised prediction …")
        # path and model may come from CLI or from config.yaml
        wav_path   = getattr(args, "path", None) or cfg.get("predict_target")
        model_path = getattr(args, "model", None) or cfg.get("model_path",
                                                             "models/speaker_count_mlp.pkl")

        if wav_path is None:
            raise SystemExit("Prediction task requires a WAV path (config 'predict_target').")

        sys.argv = [
            "run_supervised.py",
            "predict",
            wav_path,
            "--model", model_path,
            "--rate", str(cfg["rate"]),
            "--rttm_dir", cfg["rttm_dir"],
        ]
        run_supervised.cli()
    elif task == "visualize":
        print("[RUN] Visualisation …")
        # choose csv from config or latest
        csv_from_cfg = cfg.get("viz_csv")        
        sys.argv = ["visualization.py"]
        if csv_from_cfg:
            sys.argv.append(csv_from_cfg)
        # hand-off
        from src import visualization
        visualization.main()
    elif task == "cluster_scatter":
        print("[RUN] Cluster-scatter visual …")
        wav   = cfg.get("scatter_wav")
        if wav is None:
            raise SystemExit("Set scatter_wav in config.yaml (path to one .wav)")

        method = cfg.get("scatter_method", "spectral")
        rate   = str(cfg.get("scatter_rate", 4))

        # Build argv for visualization module
        sys.argv = [
            "visualization.py",   # dummy argv[0]
            wav,                  # ← first positional arg = WAV file
            "--method", method,
            "--rate",   rate,
        ]

        # Hand-off to vis module
        from src import visualization
        visualization.main_cluster_scatter()
    else:
        raise SystemExit(f"Unknown task {task}")

if __name__ == "__main__":
    main()
