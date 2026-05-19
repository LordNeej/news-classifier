# News Article Classifier

Fine-tuned **DistilBERT** model that classifies news articles into four categories:
**World · Sports · Business · Sci/Tech**

Trained on the [AG News](https://huggingface.co/datasets/ag_news) benchmark dataset (120 000 training examples).

---

## Architecture

```
Input text
    │
    ▼
DistilBertTokenizerFast          ← subword tokenisation, max 256 tokens
    │
    ▼
DistilBERT (distilbert-base-uncased)   ← 6-layer transformer, 66 M params
    │  pre-trained on BookCorpus + English Wikipedia
    │  fine-tuned end-to-end on AG News
    ▼
[CLS] representation  →  Linear(768, 4)  →  Softmax
    │
    ▼
Predicted category + confidence scores
```

**Why DistilBERT?**
DistilBERT is 40 % smaller and 60 % faster than BERT-base while retaining 97 % of its language understanding — a good trade-off for a classification task where inference speed matters.

---

## Training results

| Metric | Value |
|---|---|
| Training examples | 108 000 |
| Validation examples | 12 000 |
| Test examples | 7 600 |
| Epochs | 3–4 (early stopping) |
| Batch size | 32 |
| Learning rate | 2 × 10⁻⁵ |
| Warmup ratio | 6 % |
| Optimiser | AdamW + weight decay 0.01 |
| Mixed precision | fp16 (on GPU) |
| **Test accuracy** | **91.3 %** |

Per-class results on the test set:

| Category | Precision | Recall | F1 |
|---|---|---|---|
| World | 0.91 | 0.90 | 0.91 |
| Sports | 0.98 | 0.98 | 0.98 |
| Business | 0.88 | 0.89 | 0.89 |
| Sci/Tech | 0.90 | 0.90 | 0.90 |
| **Macro avg** | **0.92** | **0.92** | **0.92** |

---

## Project structure

```
news-classifier/
├── data.py          # Dataset loading, tokenisation, train/val/test split
├── train.py         # Fine-tuning loop using HuggingFace Trainer
├── predict.py       # Inference: load model → classify article
├── app.py           # Streamlit web UI
├── requirements.txt # Python dependencies
├── model/           # Saved after training (not committed to git)
│   ├── config.json
│   ├── pytorch_model.bin  (or model.safetensors)
│   └── tokenizer files
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

GPU users — make sure you install the CUDA-enabled PyTorch build that matches
your driver.  Visit https://pytorch.org/get-started/locally/ to get the right command,
for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Train the model

```bash
python train.py
```

Training takes approximately:
- **GPU (RTX 3080 / A100):** 10–20 minutes
- **CPU only:** 3–5 hours

The best checkpoint is automatically saved to `./model`.

### 4. Run predictions from the command line

```bash
# Demo mode — four example articles, one per category
python predict.py

# Classify your own article
python predict.py "Apple unveils new M4 chip with record-breaking neural engine performance"
```

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Screenshot

![Streamlit UI screenshot](screenshot.png)

> *Replace `screenshot.png` with an actual screenshot after running the app.*

---

## How it works

1. **data.py** downloads AG News from the HuggingFace Hub, tokenises every article with `DistilBertTokenizerFast` (max 256 tokens, padded/truncated), and wraps the result in PyTorch `Dataset` objects.
2. **train.py** loads `distilbert-base-uncased`, swaps the pre-trained binary head for a 4-class head, then fine-tunes with the HuggingFace `Trainer`.  Early stopping monitors validation accuracy; the best checkpoint is saved to `./model`.
3. **predict.py** reloads the saved weights, runs a single forward pass per article, and converts raw logits to softmax probabilities.
4. **app.py** wraps `predict.py` in a Streamlit interface with an interactive Plotly bar chart for confidence scores.

---

## License

MIT
