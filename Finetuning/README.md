# 📄 Table VQA Fine-Tuning with LLaMA-3 8B

This repository provides a full pipeline for **Supervised Fine-Tuning** of the **Meta-LLaMA-3-8B-Instruct** model on the **Table Visual Question Answering (Table VQA)** task. It supports training on HTML-rendered tables with natural language questions and answers in OTSL format and evaluates performance using robust metrics like **Exact Match**, **Levenshtein Accuracy**, and **FinTabNet-style Relieved Accuracy**.

---

## What is This Repository For?

This repository is designed for researchers and engineers working on **document foundation models**, especially in the subdomain of **table understanding**. The goal is to train large language models to answer natural language questions based on **tabular data**.

We use a structured input format (HTML, Markdown, or plain text), and fine-tune a LLaMA-3 model using supervised data. Evaluation includes both strict and relaxed accuracy metrics, and predictions are stored in JSON format for further analysis.


### 📦 Installation

###  Manual Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SaikiranKas/TableVQA.git
   cd TableVQA

2.Install dependencies:
  ```bash
  pip install -r requirements.txt
##🐳 Docker Setup
1.Build Docker image:
```bash
docker build -t llama8b -f src/model/Dockerfile .
2.Run Docker container:
```bash
docker run --rm --gpus '"device=0"' -it llama8b bash
3.Log in to HuggingFace to download Meta's LLaMA model:
```bash
huggingface-cli login
---
##Training
Train the model on HTML-structured questions and answers.
##Evaluation Metrics
You can run evaluation using various scripts provided:
```bash
# Standard metrics
python src/model/llama8baccuracy.py

# FinTabNet-style relaxed accuracy
python src/model/llama8bfintabnetaccuracy.py

# Format-specific
python src/model/llama8bplaintextaccuracy.py
python src/model/llama8bmarkdownaccuracy.py
python src/model/llama8bhtmlaccuracy.py

##Metric Details
| Metric                            | Description                                                                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Exact Match (EM)**              | Checks for exact string equality between prediction and ground truth (case- and whitespace-insensitive).                                     |
| **Levenshtein Accuracy**          | Measures how many edits are needed to convert predicted answer to ground truth. Normalized as:<br> `1 - (Levenshtein Distance / Max Length)` |
| **Relieved Accuracy (FinTabNet)** | Allows for formatting and minor textual variations but penalizes incorrect content. Useful for evaluating table-based answers.               |
---
##🧾 Prediction Output Format
After evaluation, predictions are saved as JSON files like predictions.json.
Each entry looks like:
```bash
{
  "table_id": "tbl_001",
  "question": "What is the discount rate in 2012?",
  "predicted_answer": "3.65%",
  "ground_truth": "3.65%"
}



