# 🧠 Small GPT from Scratch (Built with PyTorch)

This project is my hands-on implementation of a **Small Language Model (GPT-style)** built **completely from scratch using PyTorch**.  
It’s inspired by the original Transformer architecture described in *“Attention is All You Need”*, and trained on the **TinyStories** dataset.

> 🎯 **Goal:** Understand the inner workings of GPT models — from tokenization to self-attention — and build a fully functional small transformer model capable of generating text.

---

## 🚀 Highlights

✅ Implemented key GPT components manually:
- Token Embeddings and Positional Encodings  
- Multi-Head Causal Self-Attention  
- Feedforward (MLP) + Residual + LayerNorm blocks  
- Dropout & Weight Tying  
- Custom training loop with mixed precision, gradient accumulation, and cosine learning rate scheduling  

✅ Trained on **TinyStories** dataset for efficient experimentation.  
✅ Generates grammatically correct and contextually coherent short stories.  
✅ Fully reproducible architecture — no external pre-trained models used.  

---

## 📚 Dataset

**Dataset:** [TinyStories (Hugging Face)](https://huggingface.co/datasets/roneneldan/TinyStories)  
Each record is a short, simple story — perfect for training a small model to learn language structure without massive compute.

**Data Processing Steps:**
1. Tokenized text into subword units.
2. Created binary training (`train.bin`) and validation (`validation.bin`) files for efficient memory-mapped loading.
3. Used batches of token sequences for autoregressive next-token prediction.

---

## 🏗️ Model Architecture

| Component | Description |
|------------|--------------|
| **Embedding Layer** | Token + Positional embeddings combined |
| **Transformer Blocks** | Stacked residual blocks with LayerNorm, Multi-Head Self-Attention, and MLP |
| **Attention Heads** | Enable context learning and token dependencies |
| **Output Head** | Linear projection to vocabulary logits |
| **Loss Function** | Cross Entropy with label smoothing |
| **Optimizer** | AdamW with CosineAnnealingWarmRestarts scheduler |

**Configuration Example:**
```python
GPTConfig(
    block_size=128,
    vocab_size=50304,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.1,
    bias=True
)
```
---
## 🧮 Training Details

| Parameter | Value |
|------------|-------|
| **Learning Rate** | `1e-4 → 1e-5` (cosine schedule) |
| **Optimizer** | `AdamW (β2=0.95, weight_decay=0.1)` |
| **Batch Size** | 32 (gradient accumulation = 32) |
| **Mixed Precision** | Enabled (`bfloat16` / `float16`) |
| **Max Sequence Length** | 128 tokens |
| **Dropout** | 0.1 |
| **Device** | NVIDIA T4 (Kaggle GPU) |

> Training loss converged to ~4.8 after ~10K iterations.  
> The model starts generating meaningful, structured English text.

---

## 🧩 Inference Example

```python
import torch, json, pickle
from model_architecture import GPT, GPTConfig

# Load config
with open("Model/config.json") as f:
    config = GPTConfig(**json.load(f))

# Load model
model = GPT(config)
model.load_state_dict(torch.load("Model/small_gpt_weights.pt", map_location="cpu"))
model.eval()

# Load tokenizer
with open("Model/tokenizer.pkl", "rb") as f:
    enc = pickle.load(f)

# Generate text
prompt = "Once upon a time"
context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0)
out = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=50)

```


---

## 📬 Connect With Me
  💼 **LinkedIn**: [linkedin.com/in/nisargjani](https://linkedin.com/in/nisargjani)
