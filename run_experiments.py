import os
import sys
import time
from typing import Dict, Any
import pandas as pd

from src.utils import set_seed, load_config, setup_logger, get_device
from src.data import (
    load_raw_or_cleaned_data,
    create_stratified_splits,
    build_pytorch_datasets,
    clean_text,
)
from src.baseline import NaiveBayesBaseline
from src.model import get_model_and_tokenizer, count_parameters
from src.train import train_distilbert
from src.evaluate import (
    evaluate_distilbert,
    plot_confusion_matrix,
    plot_metrics_comparison,
    save_metrics_summary,
)
from src.inference import BlogClassifier

logger = setup_logger("run_experiments")


def print_comparison_table(nb_metrics: Dict[str, Any], bert_metrics: Dict[str, Any]) -> None:
    """Prints a comparison summary table to stdout."""
    headers = ["Metric", "Multinomial Naive Bayes", "DistilBERT (Fine-Tuned)", "Delta (Improvement)"]
    rows = [
        [
            "Accuracy",
            f"{nb_metrics['accuracy'] * 100:.2f}%",
            f"{bert_metrics['accuracy'] * 100:.2f}%",
            f"{(bert_metrics['accuracy'] - nb_metrics['accuracy']) * 100:+.2f}%",
        ],
        [
            "Macro Precision",
            f"{nb_metrics['precision_macro'] * 100:.2f}%",
            f"{bert_metrics['precision_macro'] * 100:.2f}%",
            f"{(bert_metrics['precision_macro'] - nb_metrics['precision_macro']) * 100:+.2f}%",
        ],
        [
            "Macro Recall",
            f"{nb_metrics['recall_macro'] * 100:.2f}%",
            f"{bert_metrics['recall_macro'] * 100:.2f}%",
            f"{(bert_metrics['recall_macro'] - nb_metrics['recall_macro']) * 100:+.2f}%",
        ],
        [
            "Macro F1-Score",
            f"{nb_metrics['macro_f1'] * 100:.2f}%",
            f"{bert_metrics['macro_f1'] * 100:.2f}%",
            f"{(bert_metrics['macro_f1'] - nb_metrics['macro_f1']) * 100:+.2f}%",
        ],
        [
            "Weighted F1-Score",
            f"{nb_metrics['weighted_f1'] * 100:.2f}%",
            f"{bert_metrics['weighted_f1'] * 100:.2f}%",
            f"{(bert_metrics['weighted_f1'] - nb_metrics['weighted_f1']) * 100:+.2f}%",
        ],
        [
            "Training Time",
            f"{nb_metrics.get('training_time_seconds', 0.0):.2f} sec",
            f"{bert_metrics.get('training_time_seconds', 0.0):.2f} sec ({bert_metrics.get('training_time_seconds', 0.0)/60.0:.2f} min)",
            f"{bert_metrics.get('training_time_seconds', 0.0) - nb_metrics.get('training_time_seconds', 0.0):+.2f} sec",
        ],
        [
            "Inference Latency / Sample",
            f"{nb_metrics['latency_ms_per_sample']:.2f} ms",
            f"{bert_metrics['latency_ms_per_sample']:.2f} ms",
            f"{bert_metrics['latency_ms_per_sample'] - nb_metrics['latency_ms_per_sample']:+.2f} ms",
        ],
    ]

    # If tabulate is not available, print formatted text table
    print("\n" + "=" * 80)
    print("EXPERIMENTAL EVALUATION & MODEL COMPARISON SUMMARY")
    print("=" * 80)
    col_widths = [26, 26, 26, 20]
    header_line = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * 98)
    for row in rows:
        print("".join(f"{str(c):<{w}}" for c, w in zip(row, col_widths)))
    print("=" * 80 + "\n")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device()
    logger.info(f"Initialized experiment with seed={seed} on device={device}")

    artifacts_dir = cfg.get("artifacts", {}).get("output_dir", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load Data
    data_cfg = cfg["data"]
    df = load_raw_or_cleaned_data(
        data_path=data_cfg.get("raw_data_path", "blogs.csv"),
        text_column=data_cfg.get("text_column", "Data"),
        label_column=data_cfg.get("label_column", "Labels"),
    )
    logger.info(f"Loaded dataset: {df.shape[0]} samples across {df[data_cfg['label_column']].nunique()} categories.")

    # 2. Stratified Splits (70% Train, 15% Val, 15% Test)
    train_df, val_df, test_df, label2id, id2label = create_stratified_splits(
        df=df,
        text_column=data_cfg["text_column"],
        label_column=data_cfg["label_column"],
        train_ratio=float(data_cfg.get("train_ratio", 0.70)),
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        test_ratio=float(data_cfg.get("test_ratio", 0.15)),
        seed=seed,
    )
    logger.info(
        f"Splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Stratified balance verified)."
    )

    # 3. Train & Evaluate Naive Bayes Baseline
    base_cfg = cfg.get("baseline", {})
    logger.info("Training Multinomial Naive Bayes Baseline (TF-IDF 5000 features)...")
    nb_baseline = NaiveBayesBaseline(
        max_features=int(base_cfg.get("max_features", 5000)),
        alpha=float(base_cfg.get("alpha", 1.0)),
    )
    train_nb_texts = [clean_text(text) for text in train_df[data_cfg["text_column"]].tolist()]
    test_nb_texts = [clean_text(text) for text in test_df[data_cfg["text_column"]].tolist()]
    nb_baseline.fit(
        train_nb_texts,
        train_df[data_cfg["label_column"]].tolist(),
    )
    nb_metrics = nb_baseline.evaluate(
        test_nb_texts,
        test_df[data_cfg["label_column"]].tolist(),
    )
    logger.info(
        f"Naive Bayes Test Results -> Accuracy: {nb_metrics['accuracy']*100:.2f}%, Macro-F1: {nb_metrics['macro_f1']*100:.2f}%"
    )

    # 4. Initialize Pretrained DistilBERT
    model_cfg = cfg["model"]
    model_name = model_cfg.get("pretrained_model_name", "distilbert-base-uncased")
    num_labels = len(label2id)
    logger.info(f"Loading pretrained architecture: {model_name} with {num_labels} classification classes...")
    model, tokenizer = get_model_and_tokenizer(
        model_name=model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    param_counts = count_parameters(model)
    logger.info(
        f"Model parameters: Total={param_counts['total_parameters']:,}, Trainable={param_counts['trainable_parameters']:,}"
    )

    # 5. Tokenize and Build PyTorch Datasets
    logger.info(f"Tokenizing datasets with max_length={data_cfg.get('max_length', 128)}...")
    train_dataset, val_dataset, test_dataset = build_pytorch_datasets(
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        text_column=data_cfg["text_column"],
        label_column=data_cfg["label_column"],
        label2id=label2id,
        max_length=int(data_cfg.get("max_length", 128)),
    )

    # 6. Fine-tune DistilBERT with Hugging Face Trainer
    training_cfg = cfg["training"]
    trainer, train_summary = train_distilbert(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_cfg=training_cfg,
        seed=seed,
    )

    # 7. Evaluate DistilBERT on Held-Out Test Set
    logger.info("Evaluating fine-tuned DistilBERT on test set...")
    bert_metrics = evaluate_distilbert(
        model=model,
        test_dataset=test_dataset,
        id2label=id2label,
        batch_size=int(training_cfg.get("per_device_eval_batch_size", 32)),
        device=device,
    )
    bert_metrics["training_time_seconds"] = train_summary["training_time_seconds"]
    logger.info(
        f"DistilBERT Test Results -> Accuracy: {bert_metrics['accuracy']*100:.2f}%, Macro-F1: {bert_metrics['macro_f1']*100:.2f}%"
    )

    # 8. Visualizations and Artifacts
    class_names = [id2label[i] for i in sorted(id2label.keys())]

    plot_confusion_matrix(
        cm=nb_metrics["confusion_matrix"],
        class_names=class_names,
        save_path=os.path.join(artifacts_dir, "confusion_matrix_nb.png"),
        title="Confusion Matrix: Multinomial Naive Bayes (TF-IDF)",
    )

    plot_confusion_matrix(
        cm=bert_metrics["confusion_matrix"],
        class_names=class_names,
        save_path=os.path.join(artifacts_dir, "confusion_matrix_distilbert.png"),
        title="Confusion Matrix: DistilBERT (Fine-Tuned)",
    )

    plot_metrics_comparison(
        nb_metrics=nb_metrics,
        distilbert_metrics=bert_metrics,
        save_path=os.path.join(artifacts_dir, "metrics_comparison.png"),
    )

    save_metrics_summary(
        nb_metrics=nb_metrics,
        distilbert_metrics=bert_metrics,
        output_dir=artifacts_dir,
    )

    # 9. Print Comparison Table
    print_comparison_table(nb_metrics, bert_metrics)

    # 10. Sample Inference Verification
    logger.info("Verifying inference pipeline with sample blog excerpts...")
    classifier = BlogClassifier(
        model_dir=training_cfg.get("output_dir", "saved_models/distilbert_blogs"),
        max_length=int(data_cfg.get("max_length", 128)),
        device=device,
    )
    sample_texts = [
        "NASA announces new deep space rover exploration mission targeting lunar orbital trajectories and satellite telemetry.",
        "The federal government debate over constitutional firearm regulations and second amendment rights continues in congress.",
        "New graphics driver released fixing hardware acceleration bugs on X11 display servers and openGL render pipelines.",
    ]
    sample_preds = classifier.predict_batch(sample_texts, top_k=3)
    print("\n--- SAMPLE INFERENCE VERIFICATION ---")
    for sp in sample_preds:
        print(f"Text:       '{sp['text_snippet']}'")
        print(f"Predicted:  {sp['predicted_category']} (Confidence: {sp['confidence']*100:.2f}%)")
        print("Top 3:     ", ", ".join(f"{p['category']} ({p['probability']*100:.1f}%)" for p in sp['top_k']))
        print("-" * 60)

    logger.info("All experiment tasks completed successfully!")


if __name__ == "__main__":
    main()
