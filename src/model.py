from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer


def get_model_and_tokenizer(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 20,
    id2label: Dict[int, str] = None,
    label2id: Dict[str, int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads pretrained DistilBERT model and subword tokenizer with configured
    multi-class sequence classification head.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    return model, tokenizer


def count_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """Returns total, trainable, and non-trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
    }

