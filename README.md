# Blog Topic Classification: TF-IDF + Naive Bayes vs Fine-Tuned DistilBERT

A 20-class blog topic classifier. It compares a TF-IDF + Multinomial Naive Bayes baseline with a fine-tuned `distilbert-base-uncased` model on the same data splits.

**Main finding:** On this small, keyword-driven dataset, both models reach about 87% macro-F1. There is no clear winner. See [Results](#4-results) and [Limitations](#7-limitations-and-next-steps).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Results](#4-results)
5. [Methodology](#5-methodology)
6. [How to Run](#6-how-to-run)
7. [Limitations and Next Steps](#7-limitations-and-next-steps)
8. [Configuration](#8-configuration)
9. [Reproducibility](#9-reproducibility)

---

## 1. Overview

| Component | Description |
|---|---|
| Baseline | TF-IDF (5,000 features) + Multinomial Naive Bayes |
| Transformer | `distilbert-base-uncased`, fine-tuned with Hugging Face `Trainer` |
| Task | 20-class, single-label topic classification |
| Metrics | Accuracy, macro precision, macro recall, macro-F1, weighted-F1, confusion matrix |
| Also measured | Training time and inference latency |

---

## 2. Dataset

File: `blogs.csv`

| Property | Value |
|---|---|
| Total samples | 2,000 |
| Categories | 20 (balanced, 100 per category) |
| Columns | `Data` (text), `Labels` (category) |
| Split | 1,400 train / 300 validation / 300 test (70/15/15, stratified, seed 42) |

The category names follow the 20 Newsgroups taxonomy.

<details>
<summary>20 category labels</summary>

`alt.atheism`, `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`, `misc.forsale`, `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`, `sci.crypt`, `sci.electronics`, `sci.med`, `sci.space`, `soc.religion.christian`, `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`, `talk.religion.misc`

</details>

---

## 3. Project Structure

```
.
├── blogs.csv                  # Dataset
├── run_experiments.py         # Full pipeline: train, evaluate, plot, save
├── predict.py                 # CLI inference
├── requirements.txt
├── configs/
│   └── config.yaml            # Hyperparameters and paths
├── src/
│   ├── utils.py               # Logger, seed, device, config loader
│   ├── data.py                # Cleaning, splits, tokenization, Dataset class
│   ├── baseline.py            # TF-IDF + Naive Bayes
│   ├── model.py               # DistilBERT loading
│   ├── train.py               # Trainer setup
│   ├── evaluate.py            # Metrics and plots
│   └── inference.py           # BlogClassifier (single and batch prediction)
├── artifacts/                 # Metrics JSON, plots, classification reports
├── saved_models/              # Fine-tuned model and tokenizer
├── EDA_Preprocessing.ipynb    # EDA (reference)
└── naive_bayes.ipynb          # Original prototype (reference)
```

---

## 4. Results

All numbers come from **one run** (seed 42) on a **300-sample test set**. One test sample is about 0.33 percentage points, so small gaps are within noise.

| Metric | Naive Bayes | DistilBERT | Δ (DistilBERT − NB) |
|---|---|---|---|
| Accuracy | 86.67% | 86.33% | −0.33 pp |
| Macro precision | 87.03% | 90.73% | +3.70 pp |
| Macro recall | 86.67% | 86.33% | −0.33 pp |
| Macro-F1 | 86.53% | 87.27% | +0.73 pp |
| Weighted-F1 | 86.53% | 87.27% | +0.73 pp |
| Training time | 0.39 s | 1,233.01 s (20.55 min) | +1,232.63 s |
| Inference latency | 0.12 ms/sample | 53.59 ms/sample | +53.47 ms |

### What this shows

- **No clear difference in quality.** Accuracy is lower for DistilBERT by one sample. Macro-F1 is higher by less than one point. Neither gap is reliable on 300 samples and one seed.
- **Large difference in cost.** DistilBERT took about 3,000 times longer to train and about 450 times longer per prediction.
- Both models ran on CPU.
- Latency is a single run with no warm-up pass. NB timing covers one batch call, and DistilBERT timing covers batched evaluation (batch size 32). Treat these as rough numbers.

### Artifacts

| File | Description |
|---|---|
| `artifacts/metrics_comparison.json` | All numeric results |
| `artifacts/metrics_comparison.png` | Metric comparison chart |
| `artifacts/confusion_matrix_nb.png` | Naive Bayes confusion matrix |
| `artifacts/confusion_matrix_distilbert.png` | DistilBERT confusion matrix |
| `artifacts/classification_report_nb.txt` | Per-class report, NB |
| `artifacts/classification_report_distilbert.txt` | Per-class report, DistilBERT |

---

## 5. Methodology

### Preprocessing (`src/data.py`)

- **Naive Bayes** uses `clean_text()`: lowercase, remove non-letter characters, remove scikit-learn English stop words.
- **DistilBERT** uses the original raw text, in both training and inference. Transformers are pretrained on natural sentences, so stop-word removal would hurt them.
- Both models use the same train/validation/test split.
- The number of classes is read from the data (`len(label2id)`), not from config.

### Naive Bayes baseline (`src/baseline.py`)

```
clean_text() → TF-IDF (max_features=5000) → Multinomial NB (alpha=1.0)
```

### DistilBERT (`src/model.py`, `src/train.py`)

```
Raw text → DistilBERT tokenizer (truncate/pad to max_length=128)
         → distilbert-base-uncased (about 66M parameters)
         → Linear classification head
         → Cross-entropy loss, fine-tuned for 3 epochs
```

| Parameter | Value |
|---|---|
| Max sequence length | 128 tokens |
| Learning rate | 3e-5 |
| Train / eval batch size | 16 / 32 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Warmup | `warmup_ratio=0.1` (handled by Trainer) |
| Best model selection | Validation macro-F1 |

---

## 6. How to Run

### Install

```bash
pip install -r requirements.txt
```

The first run downloads `distilbert-base-uncased` (about 260 MB) from Hugging Face Hub.

### Run the full pipeline

```bash
python run_experiments.py configs/config.yaml
```

This will:
1. Load `blogs.csv` and create stratified splits.
2. Train and evaluate Naive Bayes.
3. Fine-tune DistilBERT, validating after each epoch.
4. Save the best model to `saved_models/distilbert_blogs/`.
5. Evaluate both models on the test set.
6. Save metrics, plots and reports to `artifacts/`.

Trainer picks the available device automatically (GPU if present, otherwise CPU).

### Predict

```bash
python predict.py --text "Your blog text here" --top_k 3
```

Use `--json` for JSON output and `--model_dir` for a different checkpoint. Without `--text`, it opens an interactive prompt.

The output shows the predicted category, confidence, and the top-k candidates. Confidence is a softmax probability. It is not a calibrated measure of correctness.

---

## 7. Limitations and Next Steps

### Limitations

- **Single seed, 300 test samples.** The reported gaps between models are not statistically reliable.
- **Truncation.** `max_length=128` cuts most of each blog. Naive Bayes sees the full text, so the comparison favours NB.
- **Short training.** 3 epochs may be too few for 20 classes. Model confidence on hand-written test sentences was low.
- **Small training set.** About 70 samples per class suits Naive Bayes better than a 66M-parameter model.
- **Latency.** Measured once, without warm-up, and with different batching for each model.
- **Dataset.** Topics are mostly decided by keywords, which suits TF-IDF methods.

### Planned

- [ ] Multi-seed runs (3 to 5 seeds), reporting mean ± std
- [ ] Compare `max_length` of 128, 256 and 512
- [ ] Dynamic padding with `DataCollatorWithPadding`
- [ ] Error analysis: most confused class pairs for both models
- [ ] Third model: sentence embeddings + Logistic Regression
- [ ] FastAPI + Docker demo

---

## 8. Configuration

`configs/config.yaml`:

```yaml
experiment_name: "transformer_blog_classification"
seed: 42

data:
  raw_data_path: "blogs.csv"
  text_column: "Data"
  label_column: "Labels"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  max_length: 128

model:
  pretrained_model_name: "distilbert-base-uncased"

training:
  output_dir: "saved_models/distilbert_blogs"
  learning_rate: 3.0e-5
  per_device_train_batch_size: 16
  num_train_epochs: 3
  weight_decay: 0.01
  warmup_ratio: 0.1
  metric_for_best_model: "macro_f1"
  save_total_limit: 1

baseline:
  max_features: 5000
  alpha: 1.0

artifacts:
  output_dir: "artifacts"
```

---

## 9. Reproducibility

| Item | Value |
|---|---|
| Random seed | 42 (Python, NumPy, PyTorch) |
| Python | 3.11 |
| PyTorch | 2.10.0+cpu |
| Transformers | 5.2.0 |
| Pretrained model | `distilbert-base-uncased` |
| Hardware | CPU only |

Numbers in `artifacts/metrics_comparison.json` come from the recorded run.
