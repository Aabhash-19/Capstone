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

def generate_benchmark_dataframe(results_dict: dict) -> pd.DataFrame:
    """
    Transforms dictionary of model evaluations into a standardized benchmark DataFrame.
    """
    rows = []
    for model_name, m in results_dict.items():
        rows.append({
            "Model / Architecture": model_name,
            "Accuracy": m["Accuracy"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1 Score": m["F1 Score"],
            "FNR": m["FNR"],
            "AUC Score": m["AUC Score"]
        })
    df = pd.DataFrame(rows)
    return df

def plot_multi_model_roc(models_roc_dict: dict, save_path: str = "multi_model_roc.png"):
    """
    Plots combined ROC curves for all benchmarked model architectures.
    Accepts models_roc_dict where keys are model names and values are (y_true, y_proba) tuples.
    """
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(models_roc_dict))
    
    for (name, (y_true, y_proba)), color in zip(models_roc_dict.items(), palette):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc_val:.3f})")
        
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='No Skill (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Model ROC Comparison (Literature-Guided Architectures)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Multi-Model ROC plot to {save_path}")


def plot_model_comparison_bars(results_df: pd.DataFrame, save_path: str = "model_comparison_bars.png"):
    """Plots comparative bar chart across models for Accuracy, F1, and AUC."""
    df_melted = pd.melt(
        results_df,
        id_vars=["Model / Architecture"],
        value_vars=["Accuracy", "F1 Score", "AUC Score"],
        var_name="Metric",
        value_name="Score"
    )
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_melted,
        x="Model / Architecture",
        y="Score",
        hue="Metric",
        palette=["#2b5c8f", "#d95f02", "#7570b3"]
    )
    plt.title("Comparative Performance Across Model Architectures", fontsize=14, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Architecture", fontsize=12)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.ylim(0.5, 1.05)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    
    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=7.5, xytext=(0, 2),
                        textcoords='offset points')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Model Comparison Bars to {save_path}")

