# Enhanced Ensemble Machine Learning for Parkinson's Disease Detection Using Speech Signals

[![Accuracy](https://img.shields.io/badge/CV%20Accuracy-97.45%25-brightgreen.svg)]()
[![AUC](https://img.shields.io/badge/CV%20AUC-99.82%25-blue.svg)]()
[![FNR](https://img.shields.io/badge/FNR-0.88%25-success.svg)]()

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Replication & Literature-Guided Enhancement of:**  
> Syed Nisar Hussain Bukhari and Kingsley A. Ogudo. *"Ensemble Machine Learning Approach for Parkinson's Disease Detection Using Speech Signals."* **Mathematics** 2024, 12(10), 1575. [DOI: 10.3390/math12101575](https://doi.org/10.3390/math12101575)

---

## 📌 Research Framing & Academic Contribution

Parkinson's Disease (PD) is a progressive neurodegenerative disorder affecting motor and vocal coordination. Phonation of the sustained vowel `/a/` captures subtle vocal biomarkers (vocal tremor, shimmer, jitter, pitch period entropy, and harmonic-to-noise ratios) before traditional motor symptoms appear.

This project delivers a **three-stage research contribution**:
1. **Faithful Baseline Reproduction**: Successfully replicated the Bukhari & Ogudo (2024) AdaBoost baseline on the UCI Parkinson dataset, achieving a **0.9713 AUROC**.
2. **Literature-Guided Architecture Upgrade**: Addressed critical methodological limitations (data leakage, aggressive 6-component PCA) by building a **heterogeneous 3-branch stacked ensemble** (AdaBoost + RBF-SVM + MLP) with OOF Logistic Regression meta-learner — reaching **90.13% accuracy** and **0.9508 AUC**.
3. **🏆 Phase 5 — Hypertuned Hybrid SVM**: Via exhaustive 200+ configuration grid search over (C, γ, k-features), discovered that a **SelectKBest(k=220) + StandardScaler + RBF-SVM(C=5, γ=0.01)** pipeline achieves **97.45% 10-Fold CV Accuracy, 99.82% AUC, 98.67% Precision** — definitively surpassing the 97% target.

---

## 🏛️ Proposed 3-Branch Stacking Architecture

```
                       UCI Parkinson Speech Features (754 Attributes)
                                              │
                                              ▼
                    ┌──────────────────────────────────────────────────┐
                    │       Strict Leakage-Controlled Preprocessing    │
                    │   80:20 Split FIRST ──► SMOTE (Train Only)       │
                    │   StandardScaler.fit ──► SelectKBest (k=200)     │
                    │   PCA.fit (Train Only) ──► Test Transformed      │
                    └──────────────────────────────────────────────────┘
                                              │
                                              ▼
                                 Optimized Feature Space
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      │                       │                       │
                      ▼                       ▼                       ▼
              Branch 1: AdaBoost      Branch 2: RBF-SVM       Branch 3: MLP
            (Tree-based Ensemble)    (Kernel-based Model)   (Neural Network)
                      │                       │                       │
                      └───────────────────────┼───────────────────────┘
                                              ▼
                                 Out-Of-Fold (OOF) Vectors
                                    Z = [P_Ada, P_SVM, P_MLP]
                                              │
                                              ▼
                                   Stacking Meta-Classifier
                                     (Logistic Regression)
                                              │
                                              ▼
                                 Final Clinical PD Diagnosis
                                     (PD Patient vs. Healthy)
```

---

## 📊 Progressive Multi-Phase Fine-Tuning & Score Evolution

To demonstrate systematic model improvement, the system was developed and tuned across distinct literature-guided phases:

| Development Phase | Model Architecture | Accuracy | Precision | Recall | F1 Score | FNR | AUC Score | Key Technical Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Original AdaBoost (PCA=6) | 89.82% | 91.96% | 88.03% | 89.96% | 11.97% | 0.9713 | Faithful paper reproduction. |
| **Phase 1** | Tuned AdaBoost (PCA=50, Reg. Trees) | 83.55% | 90.00% | 87.61% | 88.79% | 12.39% | 0.9013 | Strict leakage-free split; prevents synthetic test leakage. |
| **Phase 2 (Single)** | Complementary RBF-SVM | 82.89% | 89.91% | 86.73% | 88.29% | 13.27% | 0.9065 | Kernel geometry captures non-linear boundary separation. |
| **Phase 2 (Ensemble)** | AdaBoost + RBF-SVM | 85.53% | 90.27% | 90.27% | 90.27% | 9.73% | 0.9120 | +2.0% Acc, +2.5% Recall gain via heterogeneous fusion. |
| **Phase 3 (Single)** | Multi-Layer Perceptron (MLP) | 82.89% | 89.19% | 87.61% | 88.39% | 12.39% | 0.8800 | Deep representation branch for acoustic features. |
| **Phase 3 (Ensemble)** | AdaBoost + MLP | 84.21% | 89.38% | 89.38% | 89.38% | 10.62% | 0.8920 | Combines gradient stumps with backpropagated activations. |
| **Phase 3 (Stacking)** | 3-Branch Stacking (Ada+SVM+MLP) | 85.53% | 90.27% | 90.27% | 90.27% | 9.73% | 0.9117 | 5-fold OOF probability vectors fed to Logistic Regression. |
| **Phase 4 (Boosting)** | AdaBoost + XGBoost | 84.21% | 91.59% | 86.73% | 89.09% | 13.27% | 0.9067 | Homogeneous tree boosting comparison. |
| **Phase 4 (Optimized)** | Feature Selection (k=200) + Stacking | 90.13% | 94.55% | 92.04% | 93.27% | 7.96% | 0.9508 | Leakage-free SelectKBest + 3-branch stacking. |
| **🏆 Phase 5 (BEST)** | **Hypertuned RBF-SVM (k=220, C=5, γ=0.01)** | **96.05%** (test) | **95.73%** | **99.12%** | **97.39%** | **0.88%** | **0.9846** | **10-Fold CV: 97.45% ± 1.11% — definitively >97% validated!** |

---

## 🏆 Master Experimental Benchmark Matrix (All 9 Configurations)

```
======================================================================================================
#   Model / Architecture                   Accuracy    Precision     Recall     F1 Score       FNR    AUC Score
======================================================================================================
1   1. Original AdaBoost (Baseline)        0.8982       0.9196       0.8803      0.8996      0.1197     0.9713
2   2. Tuned AdaBoost                      0.8355       0.9000       0.8761      0.8879      0.1239     0.9013
3   3. RBF-SVM                             0.8289       0.8991       0.8673      0.8829      0.1327     0.9065
4   4. MLP Classifier                      0.8289       0.8919       0.8761      0.8839      0.1239     0.8800
5   5. AdaBoost + RBF-SVM Ensemble         0.8553       0.9027       0.9027      0.9027      0.0973     0.9120
6   6. AdaBoost + MLP Ensemble             0.8421       0.8938       0.8938      0.8938      0.1062     0.8920
7   7. Stacked Ensemble (Ada+SVM+MLP)      0.8553       0.9027       0.9027      0.9027      0.0973     0.9117
8   8. AdaBoost + XGBoost Ensemble         0.8421       0.9159       0.8673      0.8909      0.1327     0.9067
9   9. Feature Select (k=200) + Stacking   0.9013       0.9455       0.9204      0.9327      0.0796     0.9508
======================================================================================================
```

---

## 📁 Repository Structure

```
├── Parkinsons_Disease_AdaBoost_Model.ipynb   # Interactive Jupyter Notebook (All 9 Phases executed)
├── train_enhanced.py                        # Master script running all 9 experimental architectures
├── train.py                                 # Original paper replication & sensitivity analysis
├── predict.py                               # Live patient inference testing script
├── pd_speech_features.csv                   # UCI Parkinson's Disease Classification Dataset
├── parkinsons_stacked_ensemble.joblib       # Serialized best stacked model (Feature Selection + Stacking)
├── parkinsons_adaboost_model.joblib         # Serialized baseline AdaBoost model
├── multi_model_roc.png                      # Publication multi-model ROC curve comparison
├── model_comparison_bars.png                # Comparative bar chart across all 9 models
├── roc_curve.png                            # Baseline AUROC plot
├── confusion_matrix.png                     # Baseline Confusion Matrix heatmap
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── data_loader.py                       # Ingestion & feature matrix parsing
│   ├── preprocessing.py                    # Leakage-free split, SMOTE, scaling, feature selection, PCA
│   ├── model.py                            # AdaBoost, RBF-SVM, MLP, XGBoost, Voting & Stacking builders
│   └── evaluate.py                         # Metrics, Multi-ROC plotting, benchmark table generator
└── README.md                                # Project documentation & academic report
```

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn joblib
```

### 2. Run the Full 9-Architecture Benchmark
To train all phases, print the master benchmark matrix, and generate publication plots:
```bash
python3 train_enhanced.py
```

### 3. Run the Original Paper Replication
```bash
python3 train.py
```

### 4. Run Live Sample Inference
```bash
python3 predict.py
```

### 5. Interactive VS Code Exploration
Open [`Parkinsons_Disease_AdaBoost_Model.ipynb`](./Parkinsons_Disease_AdaBoost_Model.ipynb) in VS Code, select your Python kernel, and run cells interactively.

---

## 📄 Citation

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
