# Contextual Modeling and Classification of Primary Emotions in Classical Indian Poetry

## 1. Project Overview

This project aims to identify and classify the dominant affective–ethical emotion expressed in classical Indian poetry written in Hindi, Tamil, and Bengali. The system combines robust traditional NLP models for classification with multilingual transformer-based semantic analysis to improve interpretability while respecting Indian aesthetic semantics.

---

## 2. Project Structure

```
Hack4Health/
│
├── round1/
│   ├── train.xlsx
│   ├── val.xlsx
│   ├── Primary_Emotions_Processed.xlsx
│
├── notebooks/
│   ├── 1_Data_Analysis.ipynb
│   ├── 2_Visualization.ipynb
│   ├── 3_Baseline_Model.ipynb
│   ├── 5_Logistic_Regression.ipynb
│
├── models/
│   ├── baseline_lr.pkl
│   ├── baseline_svm.pkl
│
├── scripts/
│   ├── verify_lr.py
│   ├── baseline_model.py
│   ├── xlmr_embeddings.py
│
├── reports/
│   ├── confusion_matrix.png
│   ├── metrics.json
│
└── Presentation1.pptx
```

---

## 3. Dataset Description
*   **Total Samples**: 4,124 poems
*   **Languages**: Hindi, Tamil, Bengali
*   **Labels**: 46 primary emotion categories
*   **Task Type**: Single-label, multi-class classification

Each poem is associated with one dominant emotion label. The dataset exhibits severe class imbalance, with a long-tailed distribution of emotion categories.

---

## 4. Data Preprocessing

Minimal, poetry-safe preprocessing was applied:
*   Unicode normalization
*   Whitespace normalization
*   No stopword removal
*   No stemming or lemmatization
*   No translation

This ensures preservation of poetic structure, symbolism, and cultural nuance.

**Additional metadata**:
*   `word_count`
*   `script_match` (language verification)

---

## 5. Feature Engineering

### 5.1 Text Representation
*   **Character-level TF-IDF (3–5 grams)**
*   Language-agnostic and robust across Indic scripts

### 5.2 Contextual Dimensions (Analysis)
Poems were analyzed using four contextual dimensions:
*   Devotion
*   Morality
*   Nature
*   Symbolism

These dimensions were used for interpretation, not for final classification.

---

## 6. Models Implemented

### 6.1 Logistic Regression (Final Classification Model)
*   **Input**: TF-IDF character n-grams
*   **Handles multi-class classification**
*   **Stable under severe class imbalance**

**Performance**:
*   Accuracy ≈ 0.34
*   Macro F1 ≈ 0.24
*   Weighted F1 ≈ 0.32

This model was selected as the final classifier due to its reliability and interpretability.

### 6.2 Linear SVM (Alternative Baseline)
*   Similar feature representation
*   Slightly higher accuracy but comparable Macro F1
*   Used for comparison and validation

### 6.3 XLM-RoBERTa (Analysis Only)
*   Used for:
    *   Semantic embedding extraction
    *   UMAP / t-SNE visualization
    *   Understanding emotion overlap

Fine-tuning was attempted but showed instability due to:
*   Extreme class imbalance
*   Limited samples per emotion

Therefore, XLM-RoBERTa was retained only for analysis and visualization, not for final classification.

---

## 7. Evaluation Metrics

Given the imbalanced dataset, evaluation focused on:
*   **Accuracy**
*   **Macro F1-score (primary metric)**
*   **Weighted F1-score**

Confusion matrices and per-class metrics were generated for detailed error analysis.

---

## 8. Visualization & Explainability

The project includes:
*   Emotion distribution plots
*   Language-wise data distribution
*   Confusion matrices
*   Semantic embedding visualization using XLM-R

These visualizations help explain why certain emotions overlap (e.g., Sadness vs Melancholy).

---

## 9. How to Run the Code

### Requirements
*   Python 3.9+
*   pandas, numpy, scikit-learn
*   matplotlib
*   transformers (for embedding analysis)
*   torch

### Example

```bash
conda activate main_nlp_env
python scripts/verify_lr.py
```

---

## 10. Conclusion

This system demonstrates that robust traditional NLP models, combined with transformer-based semantic analysis, are effective for emotion classification in culturally rich, imbalanced poetic datasets. The approach prioritizes stability, interpretability, and cultural alignment over unstable deep fine-tuning.
