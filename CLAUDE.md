# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NLP course final project at NCHU (114-2 semester): a **Chinese/English information retrieval system** (檢索系統). The system must answer queries by retrieving relevant information from a document corpus drawn from course lecture materials.

- **Grading:** 80% test results (automated), 20% written report
- **Deliverable:** GitHub repository link + report with team member IDs and names

## Repository Structure

```
final_project/
├── task/final_project.pdf      # Assignment specification
├── raw_docs/                   # Source corpus (course lecture PDFs, c0–c7)
└── CLAUDE.md
```

Implementation code does not exist yet and needs to be created.

## Expected Architecture

The retrieval pipeline should:
1. **Parse corpus** — extract text from PDFs in `raw_docs/` (bilingual: Chinese + English)
2. **Index documents** — using vector-based, probabilistic, or transformer-based representations
3. **Process queries** — accept a query string (Chinese or English)
4. **Rank & return results** — score documents against the query and return relevant passages

Course lectures cover: TF-IDF / vector models (c1), probabilistic models (c2), ML models (c4), DL / RNNs (c5), Transformers / BERT (c6–c7).

## Development Setup (to be created)

Typical setup for a Python NLP project:

```bash
pip install -r requirements.txt
python main.py --query "your question here"
```

Key libraries expected: `transformers`, `torch`, `scikit-learn`, `jieba` (Chinese tokenization), `pdfplumber` or `PyMuPDF` (PDF parsing), `numpy`.

## Evaluation

Queries come in two types (from the assignment examples):
- Factual/definition: "What is a Time-homogeneous Markov process?"
- Procedural/calculation: "How many possible combinations for Substitution Cipher?"

The system returns ranked text passages; correctness is evaluated against a hidden test set.
