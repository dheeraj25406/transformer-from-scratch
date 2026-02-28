# Transformer From Scratch (EN → IT Translation)

A complete Encoder-Decoder Transformer implementation built from scratch in PyTorch for English → Italian machine translation.

This project implements the core Transformer architecture introduced in:

> "Attention Is All You Need" – Vaswani et al.

No pretrained models were used. The architecture, masking logic, decoding strategy, and evaluation pipeline were implemented manually.

---

##  Project Overview

This project includes:

- Custom Transformer (Encoder-Decoder)
- Multi-Head Attention
- Positional Encoding
- Masked Self-Attention (Causal Masking)
- Cross-Attention
- Greedy Autoregressive Decoding
- BLEU Evaluation using SacreBLEU
- Validation Split Evaluation

Dataset used:

- OPUS Books (English → Italian)

---

##  Results

Training BLEU: **~38.5**

Validation BLEU (10% held-out split): **~37.98**

Evaluation performed using:

- SacreBLEU
- Greedy decoding
- 200 validation samples

Minimal train-validation gap indicates stable generalization.

---

##  Architecture Details

- **final_model/** – Final trained model and tokenizers
- **runs/** – Training logs and experiment tracking
- **weights/** – Intermediate checkpoints
- **transformer.py** – Full Encoder-Decoder Transformer implementation
- **train.py** – Model training pipeline
- **infer.py** – Inference / translation script
- **evaluate.py** – BLEU score evaluation
- **transformer.ipynb** – Development notebook
- **tokenizer_en.json / tokenizer_it.json** – Tokenizer files
- **README.md** – Project documentation
- **LICENSE** – License information

---

### Encoder

- Token Embedding
- Positional Encoding
- Multi-Head Self-Attention
- Feed Forward Network
- Residual Connections + LayerNorm

### Decoder

- Masked Self-Attention
- Cross-Attention (Encoder-Decoder Attention)
- Feed Forward Network
- Residual Connections + LayerNorm

### Output Layer

- Linear projection to target vocabulary size

---

##  Attention Formula

Scaled Dot Product Attention:

Attention(Q, K, V) = softmax((QKᵀ) / √dₖ) V

Multi-Head Attention:

MultiHead(Q,K,V) = Concat(head₁,...,headₕ) Wᵒ

---

##  Masking Strategy

Two masks are used:

1. Source Mask  
   Allows full attention across source tokens.

2. Target Causal Mask  
   Uses lower triangular masking (`torch.tril`)  
   Prevents decoder from attending to future tokens.

Mask is applied before softmax to prevent information leakage.

---

##  Decoding Strategy

Greedy autoregressive decoding:

- Start with `[SOS]`
- Predict next token
- Append token
- Repeat until `[EOS]` or max length

No beam search was used in current implementation.

---

---

## ⚙️ Training Details

- Device: Apple MPS
- Epochs: 20
- Batch Size: 8
- Optimizer: Adam
- Loss: Cross-Entropy
- Dataset: OPUS Books EN-IT

---

## 📈 Evaluation

Evaluation is performed on a 10% held-out split:

```python
dataset = dataset.train_test_split(test_size=0.1)

BLEU computed using:

sacrebleu.corpus_bleu(predictions, [references])
```

---

##  Key Learnings

- **Tensor shape management** in multi-head attention
- **Broadcasting behavior** in masking mechanisms
- **Importance of scaling** in dot-product attention
- **Autoregressive decoding logic** in sequence generation
- **Proper validation split** for fair model evaluation
- **BLEU metric limitations** and interpretation nuances

---

##  Future Improvements

- Implement **Beam Search decoding**
- Add **Label Smoothing** for better generalization
- Introduce **Learning Rate Warmup (Noam Scheduler)**
- Train on a **larger dataset** for improved performance
- Add **Attention visualization** for interpretability
- Compare against a **pretrained MarianMT baseline**

---

##  Why This Project Matters

This project demonstrates:

- Deep understanding of **Transformer internals**
- Ability to implement the architecture **from scratch**
- A complete **training and evaluation pipeline**
- Clear **quantitative performance reporting**
- Practical experience with real-world NLP workflows

---

##  Author

**Dheeraj Alamuri**
B.Tech (AI/ML Focus)
Interested in ML Engineering & Generative AI Systems
