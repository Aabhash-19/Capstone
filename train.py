import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_parkinson_data
from src.preprocessing import prepare_pipeline_data
from src.model import build_adaboost_model, save_trained_pipeline
from src.evaluate import (
    evaluate_model_performance, plot_roc_curve,
    plot_confusion_matrix, print_paper_comparison_table
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'pd_speech_features.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'parkinsons_adaboost_model.joblib')

def run_hyperparameter_experiments(X, y):
    """
    Replicates Tables 1, 2, 3 and Figures 3, 4, 5 from the paper:
    - Table 1 & Fig 3: Accuracy vs Number of DTs (10, 50, 100, 500)
    - Table 2 & Fig 5: Accuracy vs Tree Depth (3, 5, 7, 9)
    - Table 3 & Fig 4: Accuracy vs Learning Rate (0.1, 0.5, 1.0)
    """
    print("\n" + "="*70)
    print("RUNNING HYPERPARAMETER SENSITIVITY EXPERIMENTS (REPLICATING PAPER TABLES 1-3)")
    print("="*70)
    
    # Preprocess data once
    X_train, X_test, y_train, y_test, _, _ = prepare_pipeline_data(X, y, test_size=0.2, n_components=6, random_state=42)
    
    # 1. Experiment: Varying Number of Decision Trees (Table 1)
    print("\n--- Table 1 Replication: Impact of Number of Base Estimators (DTs) ---")
    print(f"{'No. of DT':<10} | {'Acc':<8} | {'Precision':<10} | {'Recall':<8} | {'F1 Score':<10} | {'FNR':<8}")
    print("-" * 65)
    for n_trees in [10, 50, 100, 500]:
        base_tree = DecisionTreeClassifier(max_depth=7, random_state=42)
        clf = AdaBoostClassifier(estimator=base_tree, n_estimators=n_trees, learning_rate=1.0, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"{n_trees:<10} | {acc:<8.2f} | {prec:<10.2f} | {rec:<8.2f} | {f1:<10.2f} | {fnr:<8.2f}")

    # 2. Experiment: Varying Tree Depth (Table 2)
    print("\n--- Table 2 Replication: Impact of Base Tree Depth ---")
    print(f"{'Tree Depth':<10} | {'Acc':<8} | {'Precision':<10} | {'Recall':<8} | {'F1 Score':<10} | {'FNR':<8}")
    print("-" * 65)
    for depth in [3, 5, 7, 9]:
        base_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf = AdaBoostClassifier(estimator=base_tree, n_estimators=500, learning_rate=1.0, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"{depth:<10} | {acc:<8.2f} | {prec:<10.2f} | {rec:<8.2f} | {f1:<10.2f} | {fnr:<8.2f}")

    # 3. Experiment: Varying Learning Rate (Table 3)
    print("\n--- Table 3 Replication: Impact of Learning Rate ---")
    print(f"{'LR':<10} | {'Acc':<8} | {'Precision':<10} | {'Recall':<8} | {'F1 Score':<10} | {'FNR':<8}")
    print("-" * 65)
    for lr in [0.1, 0.5, 1.0]:
        base_tree = DecisionTreeClassifier(max_depth=7, random_state=42)
        clf = AdaBoostClassifier(estimator=base_tree, n_estimators=500, learning_rate=lr, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"{lr:<10.1f} | {acc:<8.2f} | {prec:<10.2f} | {rec:<8.2f} | {f1:<10.2f} | {fnr:<8.2f}")

def main():
    print("=========================================================================")
    print("PARKINSON'S DISEASE DETECTION MODEL REPLICATION (BUKHARI & OGUDO, 2024)")
    print("=========================================================================\n")
    
    # 1. Load Dataset
    X, y, df = load_parkinson_data(DATASET_PATH)
    
    # 2. Preprocess Data (SMOTE -> StandardScaler -> 6-PCA -> 80:20 Train-Test Split)
    X_train, X_test, y_train, y_test, scaler, pca = prepare_pipeline_data(
        X, y, test_size=0.2, n_components=6, random_state=42
    )
    
    # 3. Build & Train AdaBoost Model
    model = build_adaboost_model(
        n_estimators=500,
        learning_rate=1.0,
        max_depth=7,
        random_state=42
    )
    
    print("\n[Training] Fitting AdaBoost Model on Training Data (80% split)...")
    model.fit(X_train, y_train)
    print("[Training] Model training completed successfully.")
    
    # 4. Evaluate Model Performance
    metrics, y_pred, y_proba = evaluate_model_performance(model, X_test, y_test)
    
    # 5. Display Comparison with Paper Benchmarks
    print_paper_comparison_table(metrics)
    
    # 6. Plot & Save Visualization Artifacts
    output_dir = BASE_DIR
    plot_roc_curve(y_test, y_proba, save_path=os.path.join(output_dir, "roc_curve.png"))
    plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(output_dir, "confusion_matrix.png"))
    
    # 7. Save Trained Model Pipeline
    pipeline_dict = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'metrics': metrics,
        'feature_names': list(X.columns)
    }
    save_trained_pipeline(pipeline_dict, MODEL_SAVE_PATH)
    
    # 8. Run Sensitivity Analysis (Replicating Paper Tables 1, 2, 3)
    run_hyperparameter_experiments(X, y)
    
    print("\n[Complete] Pipeline execution finished successfully!")

if __name__ == "__main__":
    main()
