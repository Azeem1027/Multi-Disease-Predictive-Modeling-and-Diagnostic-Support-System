"""
Model Evaluation, Threshold Tuning, and Metric Reporting Module.
Computes test set Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix,
and tunes decision thresholds favoring clinical sensitivity/recall.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from src.config import FIGURES_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for given probability predictions.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate base metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "threshold": float(threshold),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.80
) -> Tuple[float, Dict[str, Any]]:
    """
    Tune decision threshold to favor sensitivity/recall in a clinical setting.
    Evaluates thresholds [0.10, 0.90] to maximize F2-score (weighting recall 2x precision)
    while ensuring recall satisfies clinical safety threshold.
    """
    candidate_thresholds = np.linspace(0.10, 0.85, 76)
    best_thresh = 0.5
    best_f2 = -1.0
    best_metrics = compute_metrics(y_true, y_prob, 0.5)

    for thresh in candidate_thresholds:
        metrics = compute_metrics(y_true, y_prob, thresh)
        # Prioritize F2-score which heavily favors recall
        if metrics["recall"] >= min_recall and metrics["f2"] > best_f2:
            best_f2 = metrics["f2"]
            best_thresh = thresh
            best_metrics = metrics
        elif best_f2 < 0 and metrics["f2"] > best_f2:
            # Fallback if min_recall cannot be satisfied
            best_f2 = metrics["f2"]
            best_thresh = thresh
            best_metrics = metrics

    logger.info("Tuned threshold: %.2f (Recall: %.4f, Precision: %.4f, F1: %.4f, AUC: %.4f)",
                best_thresh, best_metrics["recall"], best_metrics["precision"],
                best_metrics["f1"], best_metrics["roc_auc"])

    return float(best_thresh), best_metrics


def plot_and_save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_name: str,
    auc_val: float
) -> Path:
    """
    Generate and save ROC Curve figure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.title(f"{disease_name.capitalize()} Disease - ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{disease_name.lower()}_roc.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_and_save_confusion_matrix(
    cm: list,
    disease_name: str,
    threshold: float
) -> Path:
    """
    Generate and save Confusion Matrix figure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ["Negative", "Positive"]
    ax.set(xticks=np.arange(cm_arr.shape[1]),
           yticks=np.arange(cm_arr.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f"{disease_name.capitalize()} Confusion Matrix (Thresh={threshold:.2f})",
           ylabel="True condition",
           xlabel="Predicted condition")

    # Annotate counts inside cells
    thresh_val = cm_arr.max() / 2.
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(cm_arr[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh_val else "black",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{disease_name.lower()}_cm.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_disease_metadata(disease_name: str, metadata: Dict[str, Any]) -> Path:
    """
    Save model metadata, feature bounds, threshold, and metrics to JSON.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = MODELS_DIR / f"{disease_name.lower()}_meta.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", out_file)
    return out_file

