import os
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import BlogDataset
from src.utils import setup_logger

logger = setup_logger("evaluate")


def evaluate_distilbert(
    model: torch.nn.Module,
    test_dataset: BlogDataset,
    id2label: Dict[int, str],
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Evaluates DistilBERT model on held-out test dataset.
    Measures accuracy, precision, recall, macro-F1, confusion matrix, and latency.
    """
    model.eval()
    model.to(device)

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds: List[int] = []
    all_labels: List[int] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    total_inference_time = time.perf_counter() - t0
    num_samples = len(all_labels)
    latency_ms = (total_inference_time / max(num_samples, 1)) * 1000.0

    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    labels_order_idx = sorted(list(id2label.keys()))
    target_names = [id2label[i] for i in labels_order_idx]

    cm = confusion_matrix(all_labels, all_preds, labels=labels_order_idx)
    report_str = classification_report(
        all_labels, all_preds, target_names=target_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0
    )

    y_true_str = [id2label[i] for i in all_labels]
    y_pred_str = [id2label[i] for i in all_preds]

    return {
        "model_name": "DistilBERT (Fine-Tuned)",
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "macro_f1": float(f1_macro),
        "precision_weighted": float(p_wt),
        "recall_weighted": float(r_wt),
        "weighted_f1": float(f1_wt),
        "inference_time_seconds": float(total_inference_time),
        "latency_ms_per_sample": float(latency_ms),
        "confusion_matrix": cm.tolist(),
        "labels_order": target_names,
        "classification_report_str": report_str,
        "classification_report_dict": report_dict,
        "y_true": y_true_str,
        "y_pred": y_pred_str,
    }


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Generates and saves a publication-quality confusion matrix heatmap."""
    cm_arr = np.array(cm)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=15, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_metrics_comparison(
    nb_metrics: Dict[str, Any],
    distilbert_metrics: Dict[str, Any],
    save_path: str,
) -> None:
    """Plots comparative bar chart of primary classification metrics and latency."""
    metrics_to_compare = [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "precision_macro"),
        ("Macro Recall", "recall_macro"),
        ("Macro F1", "macro_f1"),
    ]

    labels = [m[0] for m in metrics_to_compare]
    nb_vals = [nb_metrics[m[1]] * 100.0 for m in metrics_to_compare]
    bert_vals = [distilbert_metrics[m[1]] * 100.0 for m in metrics_to_compare]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, nb_vals, width, label="Multinomial Naive Bayes", color="#3498db")
    rects2 = ax.bar(x + width / 2, bert_vals, width, label="DistilBERT (Fine-Tuned)", color="#2ecc71")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Performance Comparison: Naive Bayes Baseline vs. DistilBERT", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Metrics comparison plot saved to {save_path}")


def save_metrics_summary(
    nb_metrics: Dict[str, Any],
    distilbert_metrics: Dict[str, Any],
    output_dir: str = "artifacts",
) -> Dict[str, Any]:
    """Persists comparison JSON and classification reports to artifacts directory."""
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "models": {
            "naive_bayes": {
                "name": nb_metrics["model_name"],
                "accuracy": nb_metrics["accuracy"],
                "precision_macro": nb_metrics["precision_macro"],
                "recall_macro": nb_metrics["recall_macro"],
                "macro_f1": nb_metrics["macro_f1"],
                "weighted_f1": nb_metrics["weighted_f1"],
                "training_time_seconds": nb_metrics.get("training_time_seconds", 0.0),
                "inference_time_seconds": nb_metrics["inference_time_seconds"],
                "latency_ms_per_sample": nb_metrics["latency_ms_per_sample"],
            },
            "distilbert": {
                "name": distilbert_metrics["model_name"],
                "accuracy": distilbert_metrics["accuracy"],
                "precision_macro": distilbert_metrics["precision_macro"],
                "recall_macro": distilbert_metrics["recall_macro"],
                "macro_f1": distilbert_metrics["macro_f1"],
                "weighted_f1": distilbert_metrics["weighted_f1"],
                "training_time_seconds": distilbert_metrics.get("training_time_seconds", 0.0),
                "inference_time_seconds": distilbert_metrics["inference_time_seconds"],
                "latency_ms_per_sample": distilbert_metrics["latency_ms_per_sample"],
            },
        },
        "improvements": {
            "accuracy_delta": float(distilbert_metrics["accuracy"] - nb_metrics["accuracy"]),
            "macro_f1_delta": float(distilbert_metrics["macro_f1"] - nb_metrics["macro_f1"]),
        },
    }

    json_path = os.path.join(output_dir, "metrics_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Comparison metrics JSON saved to {json_path}")

    # Save classification reports as readable text files
    with open(os.path.join(output_dir, "classification_report_nb.txt"), "w", encoding="utf-8") as f:
        f.write("=== Multinomial Naive Bayes Baseline Classification Report ===\n\n")
        f.write(nb_metrics["classification_report_str"])

    with open(os.path.join(output_dir, "classification_report_distilbert.txt"), "w", encoding="utf-8") as f:
        f.write("=== DistilBERT Fine-Tuned Classification Report ===\n\n")
        f.write(distilbert_metrics["classification_report_str"])

    return summary

