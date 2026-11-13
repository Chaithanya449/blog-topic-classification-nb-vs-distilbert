import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


class NaiveBayesBaseline:
    """
    Multinomial Naive Bayes classifier with TF-IDF representation,
    preserving the exact baseline methodology from naive_bayes.ipynb
    while adding systematic timing and multi-metric evaluation.
    """

    def __init__(self, max_features: int = 5000, alpha: float = 1.0):
        self.max_features = max_features
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = MultinomialNB(alpha=alpha)
        self.training_time_seconds: float = 0.0
        self.classes_: Optional[np.ndarray] = None

    def fit(self, texts: List[str], labels: List[str]) -> "NaiveBayesBaseline":
        """Fits TF-IDF vectorizer and Multinomial Naive Bayes classifier."""
        t0 = time.perf_counter()
        X_train = self.vectorizer.fit_transform(texts)
        self.model.fit(X_train, labels)
        self.training_time_seconds = time.perf_counter() - t0
        self.classes_ = self.model.classes_
        return self

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, float, float]:
        """
        Predicts labels and measures inference latency.
        Returns (predictions, total_time_sec, ms_per_sample).
        """
        t0 = time.perf_counter()
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        total_time = time.perf_counter() - t0
        latency_ms = (total_time / max(len(texts), 1)) * 1000.0
        return preds, total_time, latency_ms

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns class probabilities."""
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, test_texts: List[str], test_labels: List[str]) -> Dict[str, Any]:
        """
        Evaluates the model on test set.
        Computes accuracy, precision, recall, macro-F1, weighted-F1,
        confusion matrix, and inference latency.
        """
        preds, inf_time, latency_ms = self.predict(test_texts)

        acc = accuracy_score(test_labels, preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            test_labels, preds, average="macro", zero_division=0
        )
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
            test_labels, preds, average="weighted", zero_division=0
        )

        labels_order = sorted(list(set(test_labels)))
        cm = confusion_matrix(test_labels, preds, labels=labels_order)
        report_str = classification_report(test_labels, preds, digits=4, zero_division=0)
        report_dict = classification_report(test_labels, preds, output_dict=True, zero_division=0)

        return {
            "model_name": "Multinomial Naive Bayes (TF-IDF)",
            "accuracy": float(acc),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "macro_f1": float(f1_macro),
            "precision_weighted": float(p_wt),
            "recall_weighted": float(r_wt),
            "weighted_f1": float(f1_wt),
            "training_time_seconds": float(self.training_time_seconds),
            "inference_time_seconds": float(inf_time),
            "latency_ms_per_sample": float(latency_ms),
            "confusion_matrix": cm.tolist(),
            "labels_order": labels_order,
            "classification_report_str": report_str,
            "classification_report_dict": report_dict,
            "y_true": list(test_labels),
            "y_pred": list(preds),
        }

