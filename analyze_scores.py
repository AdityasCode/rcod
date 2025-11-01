import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

def compute_metrics(id_scores, ood_scores):
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_scores)

    # FPR@95
    fpr, tpr, thresh = roc_curve(y_true, y_scores)
    idx = np.argmax(tpr >= 0.95)
    fpr95 = fpr[idx] if idx < len(fpr) else 1.0

    # AUPR-IN / OUT
    aupr_in = average_precision_score(y_true, y_scores)
    aupr_out = average_precision_score(1 - y_true, -y_scores)

    return {
        "AUROC": auroc,
        "FPR@95": fpr95,
        "AUPR_IN": aupr_in,
        "AUPR_OUT": aupr_out
    }, (fpr, tpr), (y_true, y_scores)


def plot_results(metrics, roc, pr_data, save_dir, prefix="results"):
    os.makedirs(save_dir, exist_ok=True)

    # ROC Curve
    fpr, tpr = roc
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={metrics['AUROC']:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(save_dir, f"{prefix}_roc.png"))
    plt.close()

    # PR Curve
    y_true, y_scores = pr_data
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(rec, prec, label=f"AUPR_IN={metrics['AUPR_IN']:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curve (Inliers)")
    plt.savefig(os.path.join(save_dir, f"{prefix}_pr_in.png"))
    plt.close()

    # Histograms
    id_scores = y_scores[y_true == 1]
    ood_scores = y_scores[y_true == 0]
    plt.figure()
    plt.hist(id_scores, bins=50, alpha=0.5, label="ID")
    plt.hist(ood_scores, bins=50, alpha=0.5, label="OOD")
    plt.legend()
    plt.title("Score Distributions")
    plt.savefig(os.path.join(save_dir, f"{prefix}_hist.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_file", type=str, required=True, help="Path to ID npy file (e.g. cifar10_test.npy)")
    parser.add_argument("--ood_file", type=str, required=True, help="Path to OOD npy file (e.g. texture.npy)")
    parser.add_argument("--save_dir", type=str, default="./results/", help="Where to save plots and metrics")
    args = parser.parse_args()

    id_scores = np.load(args.id_file)
    ood_scores = np.load(args.ood_file)
    id_scores = id_scores.flatten()   # (9000,)
    ood_scores = ood_scores.flatten() # (5580,)
    #y_scores = np.concatenate([id_scores, ood_scores])

    print("ID scores shape:", id_scores.shape)
    print("OOD scores shape:", ood_scores.shape)
    metrics, roc, pr_data = compute_metrics(id_scores, ood_scores)
    print("Metrics:", metrics)

    prefix = os.path.splitext(os.path.basename(args.ood_file))[0]
    plot_results(metrics, roc, pr_data, args.save_dir, prefix=prefix)

    # Save metrics as txt
    with open(os.path.join(args.save_dir, f"{prefix}_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
