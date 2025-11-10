# landseer_entry.py
import os
import argparse
import torch
from ImageOD.score_datasets import save_scores
from config_model import config

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

    # Instantiate model if needed by RCOD
    model = config()

    os.makedirs(args.output, exist_ok=True)

    if not torch.cuda.is_available():
        print("[RCOD] WARNING: CUDA not available.")

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

    print("[RCOD] Completed. Results written to:", args.output)

if __name__ == "__main__":
    main()
