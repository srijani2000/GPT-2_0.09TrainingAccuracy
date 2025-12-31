
# 📘 Tiny Shakespeare — GPT Training (Overfitting Setup)

This repository demonstrates training a **GPT-style Transformer model from scratch** on the **Tiny Shakespeare** dataset, with the goal of **intentional overfitting** to validate model correctness, optimizer configuration, and learning rate scheduling.

The model successfully achieves a **training loss ≈ 0.09**, confirming a correct end-to-end implementation.

---

## 📂 Dataset

- **Dataset**: Tiny Shakespeare  
- **Type**: Character-level / tokenized text  
- **Purpose**: Small, low-entropy dataset ideal for:
  - Sanity checks
  - Debugging training pipelines
  - Verifying convergence behavior

---

## 🧠 Model Overview

- GPT-style **decoder-only Transformer**
- Causal self-attention
- LayerNorm + residual connections
- No encoder (autoregressive language model)

⚠️ **Dropout is disabled (0.0)** to allow fast overfitting.

---

## ⚙️ Training Configuration

### Batch & Sequence
```
Batch size (B)   = 8
Context length   = 256 tokens
```

---

### Optimization
```
Optimizer         = AdamW (fused on CUDA)
Betas             = (0.9, 0.95)
Weight Decay      = 0.1
Gradient Clipping = 0.3
```

Weight decay is applied **only to 2D parameters** (weights), while biases and LayerNorm parameters are excluded.

---

### Learning Rate Schedule

A **warmup + cosine decay** schedule is used.

```
Max steps     = 6000
Warmup steps  = 300   (5% of total steps)

Max LR        = 6e-4
Min LR        = 3e-5
```

**LR behavior:**
- Linear warmup from `0 → 6e-4`
- Smooth cosine decay down to `3e-5`
- Enables fast early learning and stable late-stage memorization

---

## 📉 Training Results

Loss values near convergence:

```
step 4466 | loss: 0.127258 | lr: 1.259e-04
step 4468 | loss: 0.119125 | lr: 1.257e-04
step 4469 | loss: 0.117057 | lr: 1.255e-04
step 4471 | loss: 0.115879 | lr: 1.253e-04
step 4478 | loss: 0.140919 | lr: 1.245e-04
step 4479 | loss: 0.099630 | lr: 1.244e-04 ✅
```

✅ **Target loss ≈ 0.09 achieved**

---

## 🧪 Key Practices That Enabled Overfitting

- Dropout disabled (`dropout = 0.0`)
- Padding tokens excluded from loss
- Deterministic / sequential data batching
- Proper gradient clipping
- Learning rate updated **every step**
- Model always in `train()` mode

---

## ✍️ Text Generation

After training, text is generated using:

- Autoregressive decoding
- Top-K sampling (`k = 50`)
- Softmax over last-token logits
- No gradient computation (`torch.no_grad()`)

This produces coherent Shakespeare-style text, confirming successful memorization.

---

## 🎯 Purpose of This Project

This project is **not meant for generalization**, but for:

- Verifying GPT implementation correctness
- Understanding optimizer & LR scheduling behavior
- Learning how loss scales with entropy
- Building intuition for Transformer training dynamics

---

## ✅ Final Outcome

- Model converges smoothly
- Loss reaches **~0.09**
- Training pipeline validated
- Ready to scale to larger datasets
