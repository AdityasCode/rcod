# landseer_entry.py
import os
import numpy as np
import argparse
from ImageOD.score_datasets import save_scores
from config_model import config  # Landseer injects this file when running
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/data")
    parser.add_argument("--output", default="/output")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--p-train", type=float, default=0.03)
    parser.add_argument("--net", type=str, default="resnet20_32x32")
    parser.add_argument("--id-dataset", type=str, default="cifar10")
    parser.add_argument("--ood-dataset", type=str, default="texture")
    parser.add_argument("--postprocess", type=str, default="react")
    parser.add_argument("--net-ckpt-path", type=str, default="./resnet20_cifar10.ckpt")
    args = parser.parse_args()

    model = config()  # defined by Landseer when pipeline runs

    print("Loading data from:", args.input_dir)
    X_train = np.load(os.path.join(args.input_dir, "data.npy"))
    Y_train = np.load(os.path.join(args.input_dir, "labels.npy"))
    X_test = np.load(os.path.join(args.input_dir, "test_data.npy"))
    Y_test = np.load(os.path.join(args.input_dir, "test_labels.npy"))

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Run your RCOD OOD evaluation
    save_scores(argparse.Namespace(
        save_path=args.output,
        net=args.net,
        net_ckpt_path=args.net_ckpt_path,
        postprocess=args.postprocess,
        id_dataset=args.id_dataset,
        ood_dataset=args.ood_dataset,
        n_train=args.n_train,
        p_train=args.p_train,
    ))

    print("Results written to:", args.output)

if __name__ == "__main__":
    main()
