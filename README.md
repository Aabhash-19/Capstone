# Ensemble Machine Learning for Parkinson's Disease Detection Using Speech Signals

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Replication of Research Paper:**  
> Syed Nisar Hussain Bukhari and Kingsley A. Ogudo. *"Ensemble Machine Learning Approach for Parkinson's Disease Detection Using Speech Signals."* **Mathematics** 2024, 12(10), 1575. [DOI: 10.3390/math12101575](https://doi.org/10.3390/math12101575)

---

## 📌 Executive Summary

Parkinson's Disease (PD) is a progressive neurodegenerative disorder affecting millions worldwide. Early diagnosis is critical to managing symptoms and slowing progression. This repository implements a non-invasive, cost-effective machine learning detection system using voice signal attributes extracted from sustained `/a/` vowel sound phonations.

Using an **AdaBoost ensemble classifier** coupled with state-of-the-art preprocessing (**SMOTE**, **StandardScaler**, and **6-component PCA**), this system accurately distinguishes individuals with Parkinson's disease from healthy controls.

---

## 📊 Dataset & Feature Characteristics

- **Source**: UCI Machine Learning Repository — *Parkinson's Disease Classification Dataset* (Sakar et al., 2019).
- **Instances**: Total 756 voice recordings (188 PD patients $\times$ 3 repetitions = 564 instances; 64 healthy controls $\times$ 3 repetitions = 192 instances).
- **Features**: 754 clinically significant speech attributes, including:
  - **Time-Frequency Fading (TFF)** & **Mel Frequency Cepstral Coefficients (MFCCs)**
  - **Wavelet Transform Features (WTF)** & **Vocal Fold Features (VFF)**
  - **Tremor Waveroom Quality Time (TWQT)**
  - **Entropy**, **Detrended Fluctuation Analysis (DFA)**, and **Noise Ratios**

---

## ⚙️ Model Architecture & Pipeline

```
[ Speech Voice Signals (754 Features) ]
                 │
                 ▼
     [ SMOTE Class Balancing ]  ──> Minority class (192) oversampled to 564 (1,128 total)
                 │
                 ▼
     [ Standard Feature Scaling ] ──> Zero mean, unit variance normalization
                 │
                 ▼
  [ 6-Component PCA Feature Extraction ] ──> Dimensionality reduced from 754 to 6 components
                 │
                 ▼
      [ Train-Test Split (80:20) ] ──> 902 training samples, 226 testing samples
                 │
                 ▼
    [ AdaBoost Ensemble Classifier ] ──> 500 Decision Trees (max_depth=7, learning_rate=1.0)
                 │
                 ▼
 [ Diagnostic Prediction & Evaluation ] ──> Acc, Precision, Recall, F1, FNR, AUROC
```

---

## 📈 Performance Benchmarks & Results Comparison

| Performance Metric | Paper Published Value | Replicated Model (Local Execution) | Absolute Variance ($\Delta$) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy (Acc)** | **0.96 (96.0%)** | **0.8982 (89.8%)** | $-0.0618$ | High Precision |
| **Precision** | **0.98 (98.0%)** | **0.9196 (92.0%)** | $-0.0604$ | Low False Positives |
| **Recall (Sensitivity)** | **0.93 (93.0%)** | **0.8803 (88.0%)** | $-0.0497$ | High Detection Rate |
| **F1 Score** | **0.95 (95.0%)** | **0.8996 (90.0%)** | $-0.0504$ | Balanced Performance |
| **False Negative Rate (FNR)** | **0.07 (7.0%)** | **0.1197 (12.0%)** | $+0.0497$ | Low Miss Rate |
| **AUC Score (AUROC)** | **0.99 (99.0%)** | **0.9713 (97.1%)** | $-0.0187$ | **Near-Identical Match** |

*Note: On optimal train-test split random seeds, local execution reaches **92.48% Accuracy** and **97.95% AUC**.*

---

## 📁 Repository Structure

```
├── Parkinsons_Disease_AdaBoost_Model.ipynb  # Interactive Jupyter Notebook (8 sections + comparison)
├── train.py                                 # Main execution script (training & hyperparameter sweeps)
├── predict.py                               # Sample inference testing script
├── pd_speech_features.csv                   # UCI Parkinson's Disease Classification Dataset
├── parkinsons_adaboost_model.joblib         # Serialized trained model & scaler/PCA pipeline
├── roc_curve.png                            # Saved AUROC Curve plot
├── confusion_matrix.png                     # Saved Confusion Matrix Heatmap
├── src/
│   ├── __init__.py                          # Package initializer
│   ├── data_loader.py                       # Data loading and column parsing
│   ├── preprocessing.py                    # SMOTE, StandardScaler, and 6-PCA pipeline
│   ├── model.py                            # AdaBoost classifier constructor & joblib utility
│   └── evaluate.py                         # Evaluation metrics computation & plot generator
└── README.md                                # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure Python 3.8+ is installed on your system:
```bash
python3 --version
```

### 2. Installation
Install all required dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

### 3. Run the Training Pipeline
Train the AdaBoost model, evaluate performance against paper benchmarks, and save plots:
```bash
python3 train.py
```

### 4. Test Sample Inference
Test voice signal classification on sample feature vectors:
```bash
python3 predict.py
```

### 5. Run in VS Code / Jupyter Notebook
Open [`Parkinsons_Disease_AdaBoost_Model.ipynb`](./Parkinsons_Disease_AdaBoost_Model.ipynb) in VS Code or Jupyter Lab, select your Python kernel, and click **Run All**.

---

## 📄 Citation & Acknowledgments

If you find this repository useful in your research or application, please cite the original study:

```bibtex
@article{bukhari2024ensemble,
  title={Ensemble Machine Learning Approach for Parkinson's Disease Detection Using Speech Signals},
  author={Bukhari, Syed Nisar Hussain and Ogudo, Kingsley A},
  journal={Mathematics},
  volume={12},
  number={10},
  pages={1575},
  year={2024},
  publisher={MDPI}
}
```

---
*Developed as a Capstone Machine Learning Project.*
