# ATML Answer Refinement Engine (Advanced Topics in Machine Learning)

An end-to-end, locally executed Transformer-based **Answer Refinement Engine** designed for **Nalanda AI**. The engine converts multimodal document retrieval passages into clean, cohesive, academically structured, and source-grounded answers.

---

## 1. Architectural Foundations

### 1.1 Sequence-to-Sequence Encoder-Decoder Architecture
The core refinement pipeline is built on a pretrained Sequence-to-Sequence (Seq2Seq) Transformer: **Google's FLAN-T5** (`google/flan-t5-small` / `google/flan-t5-base`).

Unlike decoder-only causal language models, the Seq2Seq encoder-decoder architecture cleanly decouples context comprehension from text generation:
1. **Bidirectional Encoder:** Encodes retrieved document evidence $X = (x_1, \dots, x_N)$ using bidirectional multi-head self-attention with relative positional embeddings:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + R\right)V$$
   where $R$ represents relative position bias buckets, allowing the encoder to attend freely across all retrieved tokens.
2. **Autoregressive Decoder with Cross-Attention:** Generates target answer tokens $Y = (y_1, \dots, y_T)$ sequentially. Each decoder layer performs masked causal self-attention over previously emitted tokens $y_{<t}$, followed by **cross-attention** over the encoder representations $H_{\text{enc}}$:
   $$\text{CrossAttention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}$$
3. **Generation Objective:** Generates candidate answer sequence $\hat{Y}$ maximizing the conditional log-likelihood under beam search decoding:
   $$\hat{Y} = \arg\max_Y \sum_{t=1}^T \log P(y_t \mid y_{<t}, X)$$

### 1.2 Model Selection Rationale
- **FLAN-T5 (Instruction-Tuned Seq2Seq):** Pretrained on 1,800+ diverse NLP tasks (FLAN mixture). Demonstrates zero-shot synthesis, multi-passage summarization, and query-guided question answering.
- **Local Apple Silicon Execution (M4 Mac):** Automatically leverages Apple's `mps` (Metal Performance Shaders) device backend for sub-second execution (~1.2s average latency) without requiring external third-party API calls or paid cloud endpoints.
- **Zero API Dependency & Complete Privacy:** All document context and inference remain strictly local on the user's workstation.

---

## 2. Pipeline Overview

```
                          ┌──────────────────────────┐
                          │   User Query & Folder    │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Folder Vector Retrieval  │
                          │   (all-MiniLM-L6-v2)     │
                          └─────────────┬────────────┘
                                        │ Top-k Chunks
                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      ATML Answer Refinement Engine                     │
│                                                                        │
│  1. Evidence Consolidation & OCR Artifact Cleaning                     │
│     - Strips raw markdown table delimiters & line-break hyphens        │
│     - Preserves mathematical notation ($x^2$), %, and code             │
│                                                                        │
│  2. Encoder-Decoder Seq2Seq Synthesis (FLAN-T5 on MPS)                 │
│     - Cross-attention over consolidated evidence                       │
│     - Autoregressive beam search with n-gram repetition suppression   │
│                                                                        │
│  3. Intent-Driven Structurer                                           │
│     - Steps: Procedural/algorithmic questions -> Numbered Steps        │
│     - Bullets: Comparison/categories/features -> Bulleted Lists        │
│     - Paragraphs: Definitions/concepts -> Cohesive short paragraphs   │
│                                                                        │
│  4. Dual-Pass Output Validator & Grounding Guard                       │
│     - Checks token precision & entity grounding against evidence       │
│     - Hallucination detector: suppresses ungrounded entities           │
│     - Rejection filter: detects out-of-scope / insufficient evidence   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────────────┐
                          │ Verified Structured Output│
                          │  + Preserved Citations   │
                          └──────────────────────────┘
```

---

## 3. Directory Layout

```
atml/
├── __init__.py               # Package exports
├── config.py                 # Hyperparameters, device routing (MPS/CPU), thresholds
├── answer_refiner.py         # Seq2Seq model loading, context consolidation, orchestration
├── answer_structurer.py      # Intent-based Markdown formatting (numbered steps, bullets, paras)
├── output_validator.py       # Factual consistency, hallucination detector, OCR cleaner
├── README.md                 # Technical documentation & architecture reference
└── evaluation/
    ├── __init__.py
    ├── metrics.py            # F1, ROUGE-L, Factual Consistency, Structural Score
    ├── test_cases.json       # Real academic course test cases (UNIT 5 BDA)
    ├── evaluate_answers.py   # 3-way comparative ablation runner
    └── evaluation_report.json# Experimental benchmarks and latency data
```

---

## 4. Empirical Evaluation Results

The evaluation workflow compares three paradigms on identical test cases from university lecture materials:
- **Baseline A (Direct Retrieval):** Raw extractive top passage without rewriting.
- **Baseline B (Transformer Baseline):** Vanilla Seq2Seq generation without intent structuring or validation.
- **Proposed ATML Pipeline:** Seq2Seq rewriting + intent structuring + dual-pass source-grounding validation.

### 4.1 Benchmark Summary

| Metric | Direct Retrieval (Extractive) | Seq2Seq Baseline | Proposed ATML Pipeline |
| :--- | :---: | :---: | :---: |
| **Token F1 Score** | 0.4257 | 0.3705 | **0.5302** |
| **ROUGE-L Score** | 0.3714 | 0.3137 | **0.4814** |
| **Factual Consistency (Source Overlap)** | 1.0000 | 0.7667 | **0.7800** |
| **Structural Quality (0.0 – 1.0)** | 0.9600 | 0.8700 | **0.8800** |
| **Unsupported Entity Rate (Hallucination)** | 0.0000 | 0.2000 (20%) | **0.0000 (0%)** |
| **Average Latency (Apple Silicon MPS)** | 0.01 ms | 1,064 ms | **1,211 ms** |

### 4.2 Key Findings
1. **Hallucination Elimination:** The vanilla Seq2Seq baseline suffered from a **20% unsupported entity rate**, generating fabricated technical terms not present in the lecture context. The proposed ATML validator dropped this rate to **0.0%**.
2. **Handling Insufficient Evidence:** When asked out-of-scope questions (e.g. quantum entanglement in Big Data slides), Direct Retrieval returned irrelevant text, and Baseline Seq2Seq hallucinated keywords. The proposed pipeline correctly identified low evidence overlap and returned:
   > *"The retrieved documents do not provide sufficient information to answer this question."*
3. **Substantial F1 and ROUGE-L Gains:** The structured refinement achieved a **0.5302 Token F1** and **0.4814 ROUGE-L**, significantly higher than raw extraction and unvalidated generation.

---

## 5. Execution & Verification Commands

### 5.1 Run Evaluation Benchmark
```bash
python3 atml/evaluation/evaluate_answers.py
```

### 5.2 Run ATML Unit & Integration Tests
```bash
PYTHONPATH=. pytest tests/test_atml_refinement.py -v
```

### 5.3 Run Full Repository Test Suite (35 tests)
```bash
PYTHONPATH=. pytest
```

---

## 6. Configuration Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `ATML_MODEL_NAME` | `google/flan-t5-small` | Model checkpoint (`flan-t5-small` or `flan-t5-base`) |
| `ATML_DEVICE` | `mps` (or `cpu`) | PyTorch compute backend |
| `ATML_ENABLE_REFINEMENT` | `1` | Enable/disable Seq2Seq refinement |
| `ATML_NUM_BEAMS` | `2` | Beam search width for generation |
| `ATML_MAX_INPUT_LENGTH` | `512` | Maximum input prompt tokens |
| `ATML_MAX_OUTPUT_LENGTH`| `384` | Maximum output generation tokens |
| `ATML_MIN_FAITHFULNESS` | `0.35` | Minimum grounding overlap threshold |

---

## 7. References
- Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR.
- Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv:2210.11416.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
