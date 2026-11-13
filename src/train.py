import os
import time
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
    EvalPrediction,
)
from src.data import BlogDataset
from src.utils import setup_logger

logger = setup_logger("train")


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Computes classification evaluation metrics:
    Accuracy, Macro-Precision, Macro-Recall, and Macro-F1.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
    }


def train_distilbert(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: BlogDataset,
    val_dataset: BlogDataset,
    training_cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[Trainer, Dict[str, Any]]:
    """
    Fine-tunes DistilBERT using Hugging Face Trainer with evaluation-based checkpointing.
    Measures total training wall-clock time.
    """
    output_dir = training_cfg.get("output_dir", "saved_models/distilbert_blogs")
    os.makedirs(output_dir, exist_ok=True)

    batch_size = int(training_cfg.get("per_device_train_batch_size", 16))
    epochs = int(training_cfg.get("num_train_epochs", 3))
    total_steps = (len(train_dataset) // batch_size) * epochs
    warmup_steps = int(total_steps * float(training_cfg.get("warmup_ratio", 0.1)))

    max_steps = int(training_cfg.get("max_steps", -1))
    training_args_kwargs = dict(
        output_dir=output_dir,
        learning_rate=float(training_cfg.get("learning_rate", 3e-5)),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 32)),
        num_train_epochs=epochs,
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps,
        eval_strategy=training_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True) if max_steps <= 0 else False,
        metric_for_best_model=training_cfg.get("metric_for_best_model", "macro_f1"),
        greater_is_better=training_cfg.get("greater_is_better", True),
        logging_steps=int(training_cfg.get("logging_steps", 25)),
        save_total_limit=int(training_cfg.get("save_total_limit", 1)),
        seed=seed,
        report_to="none",
    )
    if max_steps > 0:
        training_args_kwargs["max_steps"] = max_steps
        training_args_kwargs["eval_strategy"] = "no"
        training_args_kwargs["save_strategy"] = "no"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Commencing DistilBERT fine-tuning...")
    t0 = time.perf_counter()
    train_result = trainer.train()
    training_time_seconds = time.perf_counter() - t0
    logger.info(f"Training completed in {training_time_seconds:.2f} seconds ({training_time_seconds / 60.0:.2f} minutes).")

    # Save the best model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Fine-tuned model and tokenizer saved to {output_dir}")

    summary = {
        "train_loss": float(train_result.training_loss),
        "training_time_seconds": float(training_time_seconds),
        "global_step": int(train_result.global_step),
        "log_history": trainer.state.log_history,
    }

    return trainer, summary
