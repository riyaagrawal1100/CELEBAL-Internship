import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (f1_score, confusion_matrix, roc_curve, auc,
                              ConfusionMatrixDisplay)


def per_class_f1(y_true, y_pred, class_names):
    scores = f1_score(y_true, y_pred, average=None, labels=list(range(len(class_names))),
                       zero_division=0)
    return {cls: float(s) for cls, s in zip(class_names, scores)}


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm


def plot_roc(y_true_binary, scores, out_path, title="ROC curve"):
    """y_true_binary: 1 = 'changed', 0 = 'unchanged'. scores: e.g. 1 - cosine_similarity
    (higher score => more likely changed)."""
    fpr, tpr, thresholds = roc_curve(y_true_binary, scores)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic to pick an operating point
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = thresholds[best_idx]

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", zorder=5,
                label=f"Operating point (thr={best_threshold:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {
        "auc": float(roc_auc),
        "best_threshold": float(best_threshold),
        "best_tpr": float(tpr[best_idx]),
        "best_fpr": float(fpr[best_idx]),
    }
