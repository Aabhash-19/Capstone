import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def evaluate_model_performance(model, X_test, y_test):
    """
    Calculates exact classification metrics matching Equations (7)-(11) in paper:
    Accuracy, Precision, Recall, F1 Score, False Negative Rate (FNR), AUC Score.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'FNR': fnr,
        'AUC Score': auc,
        'Confusion Matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
    }
    
    return metrics, y_pred, y_proba

def plot_roc_curve(y_test, y_proba, save_path: str = "roc_curve.png"):
    """Replicates Figure 6 (AUROC curve of the tuned model)."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AdaBoost (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('AUROC Curve for Parkinson\'s Disease Detection', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved AUROC plot to {save_path}")

def plot_confusion_matrix(y_test, y_pred, save_path: str = "confusion_matrix.png"):
    """Plots heatmap of Confusion Matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy (0)', 'PD Patient (1)'],
                yticklabels=['Healthy (0)', 'PD Patient (1)'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrix to {save_path}")

def print_paper_comparison_table(metrics: dict):
    """Prints side-by-side comparison table of replicated metrics vs paper Table 4."""
    paper_targets = {
        'Accuracy': 0.96,
        'Precision': 0.98,
        'Recall': 0.93,
        'F1 Score': 0.95,
        'FNR': 0.07,
        'AUC Score': 0.99
    }
    
    print("\n" + "="*60)
    print(f"{'Performance Metric':<25} | {'Paper Value':<12} | {'Replicated Model':<15}")
    print("="*60)
    for metric_name, paper_val in paper_targets.items():
        repl_val = metrics.get(metric_name, 0.0)
        print(f"{metric_name:<25} | {paper_val:<12.2f} | {repl_val:<15.4f}")
    print("="*60 + "\n")
