import os
from typing import Dict, Any, List, Union
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data import clean_text
from src.utils import get_device, setup_logger

logger = setup_logger("inference")


class BlogClassifier:
    """
    Inference helper for single and batch blog classification
    using the fine-tuned DistilBERT checkpoint.
    """

    def __init__(
        self,
        model_dir: str = "saved_models/distilbert_blogs",
        max_length: int = 128,
        device: torch.device = None,
    ):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}. Please train model first.")

        self.device = device or get_device()
        logger.info(f"Loading inference pipeline from {model_dir} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

    def predict(
        self,
        text: str,
        top_k: int = 3,
        apply_preprocessing: bool = False,
    ) -> Dict[str, Any]:
        """
        Predicts category for a single text string.
        Returns top prediction, confidence, and top_k predictions.
        """
        results = self.predict_batch([text], top_k=top_k, apply_preprocessing=apply_preprocessing)
        return results[0]

    def predict_batch(
        self,
        texts: List[str],
        top_k: int = 3,
        apply_preprocessing: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Predicts categories for a batch of text strings.
        """
        if not texts:
            return []

        processed_texts = [clean_text(t) if apply_preprocessing else t for t in texts]

        inputs = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        results = []
        for i, original_text in enumerate(texts):
            sample_probs = probs[i].cpu().numpy()
            top_indices = sample_probs.argsort()[::-1][:top_k]

            top_predictions = [
                {
                    "category": self.id2label[int(idx)],
                    "probability": float(sample_probs[idx]),
                }
                for idx in top_indices
            ]

            results.append(
                {
                    "text_snippet": original_text[:120] + ("..." if len(original_text) > 120 else ""),
                    "predicted_category": top_predictions[0]["category"],
                    "confidence": top_predictions[0]["probability"],
                    "top_k": top_predictions,
                }
            )

        return results

