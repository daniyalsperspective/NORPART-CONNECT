# Bloom’s Taxonomy Classification using AI, ML, Deep Learning, and Open-Weight LLMs

> A reproducible research pipeline for classifying academic learning outcomes into Bloom’s Taxonomy cognitive levels using Traditional Machine Learning, Deep Learning, Transformer Encoders, Zero/Few-Shot LLMs, and QLoRA fine-tuned open-weight Large Language Models.

---

## 📌 Project Overview

This repository contains an end-to-end research pipeline for **automatic classification of Course Learning Outcomes (CLOs)** into the six cognitive levels of **Bloom’s Taxonomy**:

1. **Remember**
2. **Understand**
3. **Apply**
4. **Analyze**
5. **Evaluate**
6. **Create**

The project compares multiple modeling approaches:

* Exploratory Data Analysis (EDA)
* Traditional Machine Learning
* Deep Learning models
* Transformer encoder baselines
* Zero-shot and few-shot open-weight LLM evaluation
* QLoRA fine-tuning of instruction-tuned LLMs
* Research gap analysis and future extension toward multi-label, explainable, and human-validated educational NLP

The goal is not only to achieve high classification performance, but also to evaluate whether open-weight LLMs are practical, reproducible, and useful for educational assessment tasks.

---

## 🎯 Research Objective

The main objective of this study is to evaluate how effectively different AI models classify academic learning outcomes into Bloom’s Taxonomy levels.

The study investigates the following research questions:

### RQ1 — Baseline Performance

How well do traditional ML and deep learning models classify learning outcomes into Bloom’s cognitive levels?

### RQ2 — Open-Weight LLM Prompting

How do instruction-tuned open-source LLMs perform in zero-shot and few-shot classification settings?

### RQ3 — QLoRA Fine-Tuning

Can parameter-efficient fine-tuning improve open-weight LLM performance for Bloom’s Taxonomy classification?

### RQ4 — Practicality and Research Novelty

What are the remaining research gaps related to multi-label classification, explainability, educator validation, and cost-aware deployment?

---

## 🧠 Why Bloom’s Taxonomy?

Bloom’s Taxonomy is widely used in education to describe the cognitive complexity of learning outcomes.

| Level      | Cognitive Meaning               | Example Action Verbs            |
| ---------- | ------------------------------- | ------------------------------- |
| Remember   | Recall facts and concepts       | define, list, identify          |
| Understand | Explain ideas or concepts       | describe, explain, summarize    |
| Apply      | Use knowledge in new situations | apply, solve, demonstrate       |
| Analyze    | Break information into parts    | analyze, compare, differentiate |
| Evaluate   | Make judgments                  | evaluate, justify, critique     |
| Create     | Produce new work                | design, construct, develop      |

Manual classification of CLOs can be time-consuming and subjective. AI-based classification can support curriculum mapping, assessment design, and educational quality assurance.

---

## 📂 Dataset

The project uses the **Monash Course Learning Outcomes dataset** stored as:

```text
sample_full.csv
```

Expected columns:

```text
Learning_outcome, Remember, Understand, Apply, Analyze, Evaluate, Create
```

### Dataset Statistics

| Item                   |  Value |
| ---------------------- | -----: |
| Total rows             | 21,380 |
| Single-label rows      | 18,773 |
| Multi-label rows       |  2,607 |
| Multi-label percentage |  12.2% |
| Number of Bloom levels |      6 |

### Raw Label Distribution

| Bloom Level | Count |
| ----------- | ----: |
| Remember    | 1,185 |
| Understand  | 5,825 |
| Apply       | 6,081 |
| Analyze     | 3,459 |
| Evaluate    | 3,834 |
| Create      | 3,887 |

> Note: In the original dataset, Bloom label columns use `1.0` for assigned labels and `NaN` for non-assigned labels. For modeling, `NaN` should be converted to `0`.

---

## 🧪 Experimental Pipeline

The research pipeline follows these stages:

```text
Dataset Loading
      ↓
Exploratory Data Analysis
      ↓
Preprocessing and Label Conversion
      ↓
Train / Validation / Test Split
      ↓
Traditional ML Baselines
      ↓
Deep Learning Models
      ↓
Transformer Encoder Baselines
      ↓
Zero-Shot and Few-Shot LLM Evaluation
      ↓
QLoRA Fine-Tuning
      ↓
Evaluation and Error Analysis
      ↓
Research Gap and Novelty Analysis
```

---

## 🤖 Models Evaluated

### 1. Traditional Machine Learning

* Multinomial Naive Bayes
* Complement Naive Bayes
* Logistic Regression
* Linear SVM
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

### 2. Deep Learning

* BiLSTM
* CNN-Text
* LSTM with pretrained word embeddings

### 3. Transformer Encoder Models

* BERT
* DistilBERT
* RoBERTa
* DeBERTa

### 4. Open-Weight LLMs for Prompting

Examples of evaluated or planned LLMs:

* Mistral-7B-Instruct
* Llama-3 / Llama-3.1 Instruct
* Qwen2 / Qwen2.5 Instruct
* Gemma-2 Instruct
* Phi-3 / Phi-4 Mini Instruct
* SmolLM2 Instruct

### 5. QLoRA Fine-Tuned Models

| Model                                         | Size | Fine-Tuning Method |
| --------------------------------------------- | ---: | ------------------ |
| mistralai/Mistral-7B-Instruct-v0.3            |   7B | QLoRA 4-bit        |
| Qwen/Qwen2-7B-Instruct or Qwen2.5-7B-Instruct |   7B | QLoRA 4-bit        |
| google/gemma-2-9b-it                          |   9B | QLoRA 4-bit        |

---

## 📊 Evaluation Metrics

### Single-Label Classification Metrics

* Accuracy
* Precision
* Recall
* Macro F1-score
* Weighted F1-score
* Balanced Accuracy
* Matthews Correlation Coefficient (MCC)
* Cohen’s Kappa
* Confusion Matrix

### Multi-Label Extension Metrics

For future or extended experiments:

* Micro F1-score
* Macro F1-score
* Hamming Loss
* Jaccard Score
* Subset Accuracy
* Per-label AUROC
* Label Cardinality Error

---

## 🧬 QLoRA Fine-Tuning Setup

The QLoRA pipeline uses memory-efficient 4-bit quantization for fine-tuning large instruction-tuned models.

Typical setup:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

Recommended memory-safe settings:

```text
Quantization: 4-bit NF4
Compute dtype: bfloat16 where supported
Batch size: small
Gradient accumulation: enabled
Max sequence length: controlled
Gradient checkpointing: enabled
Model loading: one model at a time
```

---

## 💻 Hardware Used

Main fine-tuning experiments were designed for:

| Component   | Specification                          |
| ----------- | -------------------------------------- |
| GPU         | NVIDIA RTX A6000                       |
| VRAM        | 48 GB                                  |
| Fine-tuning | QLoRA 4-bit                            |
| Framework   | Hugging Face Transformers + PEFT + TRL |

For lower-end systems, use:

* Smaller encoder models such as DistilBERT or DeBERTa-base
* TF-IDF + Logistic Regression / Linear SVM
* CPU-friendly baselines
* Kaggle or Google Colab GPU for LLM fine-tuning

---

## 📁 Suggested Repository Structure

```text
bloom-taxonomy-llm-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── README.md
│   └── sample_full.csv              # Not uploaded if dataset license restricts sharing
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_traditional_ml_baselines.ipynb
│   ├── 03_deep_learning_models.ipynb
│   ├── 04_transformer_encoder_baselines.ipynb
│   ├── 05_zero_few_shot_llm_evaluation.ipynb
│   ├── 06_qlora_finetuning_mistral.ipynb
│   ├── 07_qlora_finetuning_qwen.ipynb
│   ├── 08_qlora_finetuning_gemma.ipynb
│   └── 09_error_analysis_and_research_gaps.ipynb
│
├── src/
│   ├── data_utils.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── prompts.py
│   ├── train_ml.py
│   ├── train_encoder.py
│   ├── train_qlora.py
│   └── explainability.py
│
├── configs/
│   ├── mistral_qlora.yaml
│   ├── qwen_qlora.yaml
│   └── gemma_qlora_fixed.yaml
│
├── results/
│   ├── tables/
│   ├── figures/
│   ├── predictions/
│   └── logs/
│
└── docs/
    ├── research_protocol.md
    ├── novelty_analysis.md
    └── future_work_map.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bloom-taxonomy-llm-classification.git
cd bloom-taxonomy-llm-classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
tqdm
xgboost
lightgbm
torch
transformers
accelerate
bitsandbytes
peft
trl
datasets
evaluate
sentencepiece
protobuf
shap
lime
captum
```

---

## 🔐 Hugging Face Token Setup

Do **not** hard-code Hugging Face tokens inside notebooks or Python files.

Use environment variables instead:

```bash
# Linux / macOS
export HF_TOKEN="your_huggingface_token_here"

# Windows PowerShell
setx HF_TOKEN "your_huggingface_token_here"
```

Then login safely:

```python
from huggingface_hub import login
import os

login(token=os.environ["HF_TOKEN"])
```

---

## 🚀 How to Run

### Run EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Run traditional ML baselines

```bash
jupyter notebook notebooks/02_traditional_ml_baselines.ipynb
```

### Run zero-shot / few-shot LLM evaluation

```bash
jupyter notebook notebooks/05_zero_few_shot_llm_evaluation.ipynb
```

### Run QLoRA fine-tuning

Run one model at a time to avoid memory issues:

```bash
jupyter notebook notebooks/06_qlora_finetuning_mistral.ipynb
jupyter notebook notebooks/07_qlora_finetuning_qwen.ipynb
jupyter notebook notebooks/08_qlora_finetuning_gemma.ipynb
```

---

## 📈 Current Research Findings

The current study shows that:

* Traditional ML models provide a strong baseline.
* Transformer encoder models remain highly competitive.
* Zero-shot LLMs can classify Bloom levels but may suffer from formatting and class confusion.
* Few-shot prompting improves instruction-following and label consistency.
* QLoRA fine-tuning significantly improves open-weight LLM performance.
* Mistral and Qwen-style models are strong candidates for fine-tuned Bloom classification.
* Gemma-2 requires careful configuration due to possible EOS, padding, attention, and learning-rate sensitivity.
* Single-label classification is useful but incomplete because around 12.2% of the dataset contains multi-label learning outcomes.

---

## 🔍 Research Gaps

The following gaps remain important for publication-level research:

| Gap                           | Current Status        | Priority  |
| ----------------------------- | --------------------- | --------- |
| Multi-label classification    | Partially addressed   | Very High |
| Explainability                | Missing / early stage | Very High |
| Educator validation           | Missing               | Very High |
| Cross-dataset generalization  | Missing               | High      |
| Cost-performance benchmarking | Partial               | Medium    |
| Prompt ablation               | Partial               | Medium    |
| Reproducibility package       | Needs cleanup         | High      |

---

## 🧭 Future Work

Planned extensions:

1. **Multi-label Bloom classification** using all 21,380 rows.
2. **Explainable AI** using SHAP, LIME, Integrated Gradients, or token attribution.
3. **Educator validation** with 3–5 domain experts.
4. **Cost-aware benchmarking** comparing accuracy, VRAM, latency, and training time.
5. **Cross-discipline testing** to evaluate generalization.
6. **Ambiguity detection** where the model flags uncertain CLOs for human review.
7. **Low-resource deployment** using smaller models and quantized inference.

---

## 🧠 Research Protocol Mnemonic

Use the **PLOT** protocol:

| Letter | Meaning     |
| ------ | ----------- |
| P      | Protocol    |
| L      | Literature  |
| O      | Originality |
| T      | Trust       |

> Research sirf model chalane ka naam nahi; research model ko justify, compare, explain, aur validate karne ka naam hai.

---

## ⚠️ Important Notes

* Do not upload private API keys or Hugging Face tokens.
* If dataset redistribution is restricted, provide only the dataset loading instructions.
* Always report macro F1 and class-wise F1 because the dataset is imbalanced.
* For LLM fine-tuning, load and train one model at a time to avoid GPU memory errors.
* For publication, include multi-label classification, explainability, and human validation.

---

## 📚 References

Add complete references in your final paper/repository. Suggested references include:

* Bloom, B. S. (1956). *Taxonomy of Educational Objectives*.
* Anderson, L. W., & Krathwohl, D. R. (2001). *A Taxonomy for Learning, Teaching, and Assessing*.
* Shaikh, S., Daudpotta, S. M., & Imran, A. S. (2021). Bloom’s Learning Outcomes’ Automatic Classification Using LSTM and Pretrained Word Embeddings. *IEEE Access*.
* Li, Y., and related authors. EDM 2022 work on Monash CLO / learning outcome classification.
* Recent 2024–2026 studies on Bloom classification using BERT, DistilBERT, GPT-4, transfer learning, and educational LLMs.

---

## 👤 Author

**Muhammad Daniyal**
Researcher in Artificial Intelligence, Machine Learning, Big Data Analytics, and Generative AI
GitHub: `YOUR_GITHUB_USERNAME`
LinkedIn: `YOUR_LINKEDIN_PROFILE`

---

## 📄 License

This project is intended for academic and research purposes.

Recommended license:

```text
MIT License
```

If the dataset has redistribution restrictions, keep the code open-source but do not publicly upload the dataset.

---

## ✅ Citation

If you use this repository, please cite it as:

```bibtex
@misc{daniyal2026bloomllm,
  title        = {Bloom's Taxonomy Classification using AI, ML, Deep Learning, and Open-Weight LLMs},
  author       = {Muhammad Daniyal},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Educational NLP, Bloom's Taxonomy, QLoRA, Open-Weight LLMs}
}
```

---

## ✅ Project Status

```text
Current status: Research prototype completed
Next target: Multi-label + Explainability + Educator Validation
Publication readiness: Medium, improving toward high after trust-layer implementation
```
