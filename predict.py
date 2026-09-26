import os
import sys
import numpy as np
import pandas as pd
from src.model import load_trained_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'parkinsons_adaboost_model.joblib')
DATASET_PATH = os.path.join(BASE_DIR, 'pd_speech_features.csv')

def predict_sample(sample_df: pd.DataFrame, model_path: str = None):
    """
    Predicts Parkinson's Disease status for a new voice signal sample feature vector.
    Supports both Phase 5 Champion Model (Selector) and Baseline (PCA).
    """
    if model_path is None:
        p5_path = os.path.join(BASE_DIR, 'parkinsons_phase5_hyper_svm.joblib')
        model_path = p5_path if os.path.exists(p5_path) else MODEL_SAVE_PATH

    pipeline = load_trained_pipeline(model_path)
    model = pipeline.get('model') or pipeline.get('stacking_model')
    scaler = pipeline.get('scaler')
    selector = pipeline.get('selector')
    pca = pipeline.get('pca')
    feature_names = pipeline['feature_names']
    
    # Ensure correct feature column order
    sample_features = sample_df[feature_names]
    
    # Transform sample: Scale -> SelectKBest/PCA -> Model Predict
    sample_scaled = scaler.transform(sample_features)
    if selector is not None:
        sample_transformed = selector.transform(sample_scaled)
        if pca is not None:
            sample_transformed = pca.transform(sample_transformed)
    elif pca is not None:
        sample_transformed = pca.transform(sample_scaled)
    else:
        sample_transformed = sample_scaled
        
    preds = model.predict(sample_transformed)
    probs = model.predict_proba(sample_transformed)[:, 1] if hasattr(model, "predict_proba") else None
    
    return preds, probs

def main():
    print("==========================================================")
    print("INFERENCE UTILITY: PARKINSON'S DISEASE VOICE SIGNAL TEST")
    print("==========================================================\n")
    
    p5_path = os.path.join(BASE_DIR, 'parkinsons_phase5_hyper_svm.joblib')
    active_model = p5_path if os.path.exists(p5_path) else MODEL_SAVE_PATH
    model_name = "Phase 5 Hypertuned RBF-SVM (Champion)" if active_model == p5_path else "Baseline AdaBoost"

    if not os.path.exists(active_model):
        print(f"[Error] Trained model file not found at {active_model}. Please run `python train_enhanced.py` first.")
        sys.exit(1)
        
    print(f"[Info] Active Inference Model: {model_name}\n")
    # Load 5 sample rows from the dataset for demonstration
    df = pd.read_csv(DATASET_PATH, header=1)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    sample_subset = df.sample(5, random_state=42)
    true_labels = sample_subset['class'].values
    sample_features = sample_subset.drop(columns=['class'])
    
    preds, probs = predict_sample(sample_features, active_model)
    
    print(f"{'Sample #':<10} | {'True Status':<15} | {'Predicted Status':<20} | {'PD Confidence':<15}")
    print("-" * 70)
    for i in range(len(preds)):
        true_str = "PD Patient (1)" if true_labels[i] == 1 else "Healthy (0)"
        pred_str = "PD Patient (1)" if preds[i] == 1 else "Healthy (0)"
        conf_str = f"{probs[i]*100:.2f}%" if probs is not None else "N/A"
        print(f"{i+1:<10} | {true_str:<15} | {pred_str:<20} | {conf_str:<15}")

if __name__ == "__main__":
    main()
