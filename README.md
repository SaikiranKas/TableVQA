## Table Evaluation Toolkit (TEDS/TED)

This repository provides a clean, modular toolkit to evaluate table structure predictions using **TEDS** (Tree Edit Distance-based Similarity) and **TED** (Tree Edit Distance) metrics, particularly for datasets like FinTabNet, PubTabNet, etc.

It is designed to:
- Clean noisy HTML predictions and ground truths
- Compute both structure-only (TEDS) and full-content (TED) similarity
- Handle wrapped ground truth formats (like `{"image": [...]}`)
- Provide clean CSV outputs and optional cleaned GT saves

---

## Directory Structure

```text
docfm_evaluation/
├── utils.py                 # HTML preprocessing utilities
├── evaluate.py              # Core TEDS/TED evaluation logic
├── run_evaluation.py        # CLI entry point for running evaluation
├── requirements.txt         # Required Python packages
├── metric.py                # TEDS metric implementation 
└── README.md                # Project documentation (this file)
```
---

## Installation

```bash
git clone https://github.com/Utkarsh-TIHIITB/docfm_evaluation.git
cd docfm_evaluation
pip install -r requirements.txt
```
---

## Input Format
###  Prediction File (--pred_path)
```json
[
  {
    "filename": "some_image.jpg",
    "html": "<table>...</table>"
  },
  ...
]
```
### Ground Truth File (--gt_path)
Two possible formats
 1. Flat List:
```
[
  {
    "filename": "some_image.jpg",
    "text_html_table": "<table>...</table>"
  },
  ...
]
```
2. Wrapped Format:
  ```json
{
  "image": [
    {
      "filename": "some_image.jpg",
      "text_html_table": "<table>...</table>"
    },
    ...
  ]
}
```
---
## Run Evaluation
```bash
python run_evaluation.py \
    --pred_path path/to/predictions.json \
    --gt_path path/to/ground_truth.json \
    --output_csv path/to/teds_scores.csv \
```
---
### Arguments

| Argument            | Description                                                  |
|---------------------|--------------------------------------------------------------|
| `--pred_path`       | Path to JSON file containing predictions                     |
| `--gt_path`         | Path to ground truth JSON file                               |
| `--output_csv`      | Path to save the output CSV file with average scores         |

---
## Output Format
1. Terminal Output:
```
 Matched 250 files.
 Calculating TEDS/TED: 100%|██████████| 250/250 [00:05<00:00, 18.35it/s]

 Average TEDS (structure only): 0.8936
 Average TED  (full table):     0.7001
```
2. CSV Output:

```
avg_teds,avg_ted
0.8936,0.7001
```
---
## Example
```
python run_evaluation.py \
    --pred_path model/fintabnetqa_with_otsl.json \
    --gt_path model/fintabnetqa_qa_data.json \
    --output_csv model/fintabnetqa_ted_scores.csv \
```
## Internals

- Uses **BeautifulSoup** to remove all attributes (`style`, `bbox`, etc.)
- Replaces deprecated/unsupported HTML tags and strips out unwanted ones (`<font>`, `<span>`, etc.)
- Flattens table structure by replacing `colspan` and `rowspan` with standard layout
- Wraps GT HTML with a root `<html>` tag before TEDS evaluation
- Computes both evaluation metrics:
  - **TEDS**: Structure-only comparison (`is_structure=True`)
  - **TED**: Full content comparison









