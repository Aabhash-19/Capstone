import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_parkinson_data
from src.preprocessing import prepare_pipeline_data, prepare_leakage_free_pipeline, apply_feature_selection
from src.model import (
    build_adaboost_model, build_tuned_adaboost, build_rbf_svm,
    build_mlp, build_xgboost, build_voting_ensemble,
    build_stacking_classifier, save_trained_pipeline
)
from src.evaluate import (
    evaluate_model_performance, generate_benchmark_dataframe,
    plot_multi_model_roc, plot_model_comparison_bars
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

DATASET_PATH = 'pd_speech_features.csv'
MODEL_SAVE_PATH = 'parkinsons_stacked_ensemble.joblib'

def evaluate_classifier(model, X_test, y_test, name: str):
    """Calculates standardized metrics dictionary for any classifier."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    auc = roc_auc_score(y_test, probs) if probs is not None else 0.5
    
    print(f"[{name:<32}] Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | FNR={fnr:.4f} | AUC={auc:.4f}")
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'FNR': fnr,
        'AUC Score': auc
    }, preds, probs

def main():
    print("=" * 80)
    print("ENHANCED ENSEMBLE SYSTEM: PROGRESSIVE MODEL IMPROVEMENT PIPELINE")
    print("Literature-Guided Strategy for Bukhari & Ogudo (2024) Parkinson's Detection")
    print("=" * 80 + "\n")
    
    # 1. Load Data
    X, y, df = load_parkinson_data(DATASET_PATH)
    
    results = {}
    roc_probs = {}
    
    # =========================================================================
    # BASELINE: Original Bukhari & Ogudo (2024) Replication (PCA=6)
    # =========================================================================
    print("\n" + "-"*75)
    print("BASELINE: Original AdaBoost Replication (PCA=6, 500 Trees, Depth=7)")
    print("-"*75)
    X_tr_b, X_te_b, y_tr_b, y_te_b, _, _ = prepare_pipeline_data(X, y, test_size=0.2, n_components=6, random_state=42)
    base_ada = build_adaboost_model(n_estimators=500, learning_rate=1.0, max_depth=7, random_state=42)
    base_ada.fit(X_tr_b, y_tr_b)
    m_base, _, p_base = evaluate_classifier(base_ada, X_te_b, y_te_b, "1. Original AdaBoost (Baseline)")
    results["1. Original AdaBoost (Baseline)"] = m_base
    roc_probs["1. Original AdaBoost"] = (y_te_b, p_base)

    # =========================================================================
    # STRICT LEAKAGE-CONTROLLED PREPROCESSING FOR ALL IMPROVED PHASES
    # =========================================================================
    print("\n" + "-"*75)
    print("CONFIGURING LEAKAGE-FREE EXPERIMENTAL PIPELINE (PCA=50 components)")
    print("-"*75)
    X_train, X_test, y_train, y_test, scaler, pca, _ = prepare_leakage_free_pipeline(
        X, y, test_size=0.2, n_components=50, random_state=42
    )

    # =========================================================================
    # PHASE 1: Hyperparameter & Variance Optimization (Tuned AdaBoost)
    # =========================================================================
    print("\n" + "-"*75)
    print("PHASE 1: Tuned AdaBoost (PCA=50, Regularized Trees, Lower LR)")
    print("-"*75)
    tuned_ada = build_tuned_adaboost(
        n_estimators=300, learning_rate=0.1, max_depth=4,
        min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42
    )
    tuned_ada.fit(X_train, y_train)
    m_tada, _, p_tada = evaluate_classifier(tuned_ada, X_test, y_test, "2. Tuned AdaBoost")
    results["2. Tuned AdaBoost"] = m_tada
    roc_probs["2. Tuned AdaBoost"] = (y_test, p_tada)

    # =========================================================================
    # PHASE 2: Complementary Classical Model (RBF-SVM) & 2-Model Ensemble
    # =========================================================================
    print("\n" + "-"*75)
    print("PHASE 2: Complementary Classical Model (RBF-SVM) & 2-Model Ensemble")
    print("-"*75)
    rbf_svm = build_rbf_svm(C=5.0, gamma='scale', random_state=42)
    rbf_svm.fit(X_train, y_train)
    m_svm, _, p_svm = evaluate_classifier(rbf_svm, X_test, y_test, "3. RBF-SVM")
    results["3. RBF-SVM"] = m_svm
    roc_probs["3. RBF-SVM"] = (y_test, p_svm)

    ada_svm_ens = build_voting_ensemble({'ada': tuned_ada, 'svm': rbf_svm}, voting='soft')
    ada_svm_ens.fit(X_train, y_train)
    m_ada_svm, _, p_ada_svm = evaluate_classifier(ada_svm_ens, X_test, y_test, "5. AdaBoost + RBF-SVM Ensemble")
    results["5. AdaBoost + RBF-SVM Ensemble"] = m_ada_svm
    roc_probs["5. AdaBoost + RBF-SVM"] = (y_test, p_ada_svm)

    # =========================================================================
    # PHASE 3: Neural Model (MLP) & 3-Branch Stacking Meta-Classifier
    # =========================================================================
    print("\n" + "-"*75)
    print("PHASE 3: Neural Model (MLP) & 3-Branch Stacking Meta-Classifier")
    print("-"*75)
    mlp = build_mlp(hidden_layer_sizes=(128, 64), max_iter=500, alpha=0.001, random_state=42)
    mlp.fit(X_train, y_train)
    m_mlp, _, p_mlp = evaluate_classifier(mlp, X_test, y_test, "4. MLP Classifier")
    results["4. MLP Classifier"] = m_mlp
    roc_probs["4. MLP Classifier"] = (y_test, p_mlp)

    ada_mlp_ens = build_voting_ensemble({'ada': tuned_ada, 'mlp': mlp}, voting='soft')
    ada_mlp_ens.fit(X_train, y_train)
    m_ada_mlp, _, p_ada_mlp = evaluate_classifier(ada_mlp_ens, X_test, y_test, "6. AdaBoost + MLP Ensemble")
    results["6. AdaBoost + MLP Ensemble"] = m_ada_mlp
    roc_probs["6. AdaBoost + MLP"] = (y_test, p_ada_mlp)

    stacked_ensemble = build_stacking_classifier(
        {'ada': tuned_ada, 'svm': rbf_svm, 'mlp': mlp}, cv=5
    )
    stacked_ensemble.fit(X_train, y_train)
    m_stack, _, p_stack = evaluate_classifier(stacked_ensemble, X_test, y_test, "7. Stacked Ensemble (Ada+SVM+MLP)")
    results["7. Stacked Ensemble (Ada+SVM+MLP)"] = m_stack
    roc_probs["7. Stacked (Ada+SVM+MLP)"] = (y_test, p_stack)

    # =========================================================================
    # PHASE 4: Comparative Benchmarks (Boosting & Feature Selection)
    # =========================================================================
    print("\n" + "-"*75)
    print("PHASE 4: Comparative Benchmarks (XGBoost & Feature Selection)")
    print("-"*75)
    xgb_model = build_xgboost(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    ada_xgb_ens = build_voting_ensemble({'ada': tuned_ada, 'xgb': xgb_model}, voting='soft')
    ada_xgb_ens.fit(X_train, y_train)
    m_ada_xgb, _, p_ada_xgb = evaluate_classifier(ada_xgb_ens, X_test, y_test, "8. AdaBoost + XGBoost Ensemble")
    results["8. AdaBoost + XGBoost Ensemble"] = m_ada_xgb
    roc_probs["8. AdaBoost + XGBoost"] = (y_test, p_ada_xgb)

    # Feature selection experiment: SelectKBest (k=200) -> PCA(20) -> Stacking
    X_tr_fs, X_te_fs, y_tr_fs, y_te_fs, sc_fs, pca_fs, sel_fs = prepare_leakage_free_pipeline(
        X, y, test_size=0.2, n_components=20, k_features=200, random_state=42
    )
    ada_fs = build_tuned_adaboost(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=42)
    svm_fs = build_rbf_svm(C=5.0, random_state=42)
    mlp_fs = build_mlp(hidden_layer_sizes=(128, 64), random_state=42)
    
    fs_stack = build_stacking_classifier({'ada': ada_fs, 'svm': svm_fs, 'mlp': mlp_fs}, cv=5)
    fs_stack.fit(X_tr_fs, y_tr_fs)
    m_fs, _, p_fs = evaluate_classifier(fs_stack, X_te_fs, y_te_fs, "9. Feature Select (k=200) + Stacking")
    results["9. Feature Select (k=200) + Stacking"] = m_fs
    roc_probs["9. Feature Select + Stacking"] = (y_te_fs, p_fs)

    # =========================================================================
    # PHASE 5: HYPERTUNED RBF-SVM WITH ANOVA FEATURE SELECTION (k=220)
    # =========================================================================
    print("\n" + "-"*75)
    print("PHASE 5: Hypertuned RBF-SVM with ANOVA Feature Selection (k=220, C=5, γ=0.01)")
    print("-"*75)
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.svm import SVC
    
    # Preprocessing strictly leakage-free
    X_tr_raw, X_te_raw, y_tr_raw, y_te_p5 = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr_sm, y_tr_p5 = SMOTE(random_state=42).fit_resample(X_tr_raw, y_tr_raw)
    
    sc_p5 = StandardScaler()
    X_tr_sc = sc_p5.fit_transform(X_tr_sm)
    X_te_sc = sc_p5.transform(X_te_raw)
    
    sel_p5 = SelectKBest(f_classif, k=220)
    X_tr_p5 = sel_p5.fit_transform(X_tr_sc, y_tr_p5)
    X_te_p5 = sel_p5.transform(X_te_sc)
    
    svm_p5 = SVC(C=5, gamma=0.01, kernel='rbf', probability=True, random_state=42)
    svm_p5.fit(X_tr_p5, y_tr_p5)
    
    m_p5, _, p_p5_prob = evaluate_classifier(svm_p5, X_te_p5, y_te_p5, "10. Phase 5 Hypertuned RBF-SVM")
    results["10. Phase 5: Hypertuned RBF-SVM (k=220)"] = m_p5
    roc_probs["10. Phase 5 Hypertuned SVM"] = (y_te_p5, p_p5_prob)
    
    # 10-Fold Stratified Cross-Validation on balanced training space
    cv_p5 = cross_validate(
        SVC(C=5, gamma=0.01, kernel='rbf', probability=True, random_state=42),
        X_tr_p5, y_tr_p5,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    print(f"\n[Phase 5 10-Fold CV] Accuracy:  {cv_p5['test_accuracy'].mean()*100:.2f}% ± {cv_p5['test_accuracy'].std()*100:.2f}%")
    print(f"[Phase 5 10-Fold CV] Precision: {cv_p5['test_precision'].mean()*100:.2f}% ± {cv_p5['test_precision'].std()*100:.2f}%")
    print(f"[Phase 5 10-Fold CV] Recall:    {cv_p5['test_recall'].mean()*100:.2f}% ± {cv_p5['test_recall'].std()*100:.2f}%")
    print(f"[Phase 5 10-Fold CV] F1 Score:  {cv_p5['test_f1'].mean()*100:.2f}% ± {cv_p5['test_f1'].std()*100:.2f}%")
    print(f"[Phase 5 10-Fold CV] AUC:       {cv_p5['test_roc_auc'].mean()*100:.2f}% ± {cv_p5['test_roc_auc'].std()*100:.2f}%")

    # =========================================================================
    # TABULATE FULL BENCHMARK MATRIX (ALL 10 ARCHITECTURES)
    # =========================================================================
    benchmark_df = generate_benchmark_dataframe(results)
    
    print("\n" + "=" * 85)
    print("MASTER EXPERIMENTAL BENCHMARK MATRIX (ALL 10 ARCHITECTURES)")
    print("=" * 85)
    print(benchmark_df.to_string(index=False))
    print("=" * 85 + "\n")

    # Generate publication figures
    plot_multi_model_roc(roc_probs, save_path="multi_model_roc.png")
    plot_model_comparison_bars(benchmark_df, save_path="model_comparison_bars.png")

    # Save Best Models
    save_pipeline = {
        'stacking_model': fs_stack,
        'scaler': sc_fs,
        'selector': sel_fs,
        'pca': pca_fs,
        'feature_names': list(X.columns),
        'results_matrix': benchmark_df.to_dict(orient='records')
    }
    save_trained_pipeline(save_pipeline, MODEL_SAVE_PATH)
    
    # Save Phase 5 Champion Model
    phase5_pipeline = {
        'model': svm_p5,
        'scaler': sc_p5,
        'selector': sel_p5,
        'cv_results': {
            'accuracy_mean': cv_p5['test_accuracy'].mean(),
            'accuracy_std': cv_p5['test_accuracy'].std(),
            'auc_mean': cv_p5['test_roc_auc'].mean(),
            'precision_mean': cv_p5['test_precision'].mean(),
            'recall_mean': cv_p5['test_recall'].mean(),
            'f1_mean': cv_p5['test_f1'].mean(),
        },
        'test_results': m_p5,
        'feature_names': list(X.columns)
    }
    save_trained_pipeline(phase5_pipeline, 'parkinsons_phase5_hyper_svm.joblib')
    print("[Complete] All 10 phases executed and verified successfully!")

if __name__ == "__main__":
    main()
