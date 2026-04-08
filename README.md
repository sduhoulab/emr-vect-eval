# Self-supervision and domain specific representation learning for evaluating electronic medical records: a large-scale study based on over 80,000 hospitalization records

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides the implementation for a large-scale study evaluating clinical semantic embeddings. Using 81,299 standardized inpatient records (2012–2020) from the **Cheeloo LEAD** database, we compared four strategies: direct pre-training, embedding models, self-supervised continual pre-training, and supervised fine-tuning.

---

## Repository Structure

The core implementation is divided into three functional scripts. Please follow this execution order:

### 1. Self-Supervised Domain Adaptation
This script performs domain adaptation of a pre-trained language model (RoBERTa/BERT) using unlabeled EMR admission records via Masked Language Modeling (MLM).
```bash
python self_supervised.py
```

### 2. Supervised Fine-Tuning (SFT)
This script trains the model on clinical text using the specific labeled disease categories chosen for your study. It creates a task-specific classifier based on the clinical narratives.
```bash
python supervised_classification.py
```

### 3. Downstream Evaluation & Interpretability
This script extracts clinical embeddings and runs multiple benchmarks, including:
* Unsupervised Clustering (K-Means, Agglomerative).
* Fairness Analysis across demographics (Age, Sex).
```bash
python downstream_evaluation.py
```

---

## Datasets

**Source:** Cheeloo Lifespan EHR Academic Database (Cheeloo LEAD)  
Access to the raw 81,299 hospitalization records is subject to institutional approval and data-use agreements. More details can be found on the [official portal](http://www.mhdata.sdu.edu.cn/cheeloolead.htm).

---

## Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/EMRVectEval.git](https://github.com/yourusername/EMRVectEval.git)
cd EMRVectEval

# Install required packages
pip install torch transformers datasets pandas numpy scikit-learn tqdm openpyxl joblib matplotlib
```
