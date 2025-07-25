# 📄 Table VQA Fine-Tuning with LLaMA-3 8B

This repository provides a full pipeline for **Supervised Fine-Tuning** of the **Meta-LLaMA-3-8B-Instruct** model on the **Table Visual Question Answering (Table VQA)** task. It supports training on HTML-rendered tables with natural language questions and answers in OTSL format and evaluates performance using robust metrics like **Exact Match**, **Levenshtein Accuracy**, and **FinTabNet-style Relieved Accuracy**.

---

## 🎯 What is This Repository For?

This repository is designed for researchers and engineers working on **document foundation models**, especially in the subdomain of **table understanding**. The goal is to train large language models to answer natural language questions based on **tabular data**.

We use a structured input format (HTML, Markdown, or plain text), and fine-tune a LLaMA-3 model using supervised data. Evaluation includes both strict and relaxed accuracy metrics, and predictions are stored in JSON format for further analysis.

---

## 📦 Installation

### ✅ Manual Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SaikiranKas/TableVQA.git
   cd TableVQA

