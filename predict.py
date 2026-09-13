import os
import sys
import numpy as np
import pandas as pd
from src.model import load_trained_pipeline

MODEL_SAVE_PATH = '/Users/irray/Desktop/Projects/Capstone /parkinsons_adaboost_model.joblib'
DATASET_PATH = '/Users/irray/Desktop/Projects/Capstone /pd_speech_features.csv'

def predict_sample(sample_df: pd.DataFrame, model_path: str = MODEL_SAVE_PATH):
    """
    Predicts Parkinson's Disease status for a new voice signal sample feature vector.
    
    Parameters:
    -----------
    sample_df : pd.DataFrame
        DataFrame containing 754 feature attributes.
    model_path : str
        Path to serialized model pipeline.
        
    Returns:
    --------
    predictions : np.ndarray
        Predicted class (1 = Parkinson's Disease, 0 = Healthy)
    probabilities : np.ndarray
        Prediction confidence probabilities
    """
    pipeline = load_trained_pipeline(model_path)
    model = pipeline['model']
    scaler = pipeline['scaler']
    pca = pipeline['pca']
    feature_names = pipeline['feature_names']
    
    # Ensure correct feature column order
    sample_features = sample_df[feature_names]
    
    # Transform sample: Scale -> PCA -> AdaBoost Predict
    sample_scaled = scaler.transform(sample_features)
    sample_pca = pca.transform(sample_scaled)
    
    preds = model.predict(sample_pca)
    probs = model.predict_proba(sample_pca)[:, 1] if hasattr(model, "predict_proba") else None
    
    return preds, probs

def main():
    print("==========================================================")
    print("INFERENCE UTILITY: PARKINSON'S DISEASE VOICE SIGNAL TEST")
    print("==========================================================\n")
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[Error] Trained model file not found at {MODEL_SAVE_PATH}. Please run `python train.py` first.")
        sys.exit(1)
        
    # Load 5 sample rows from the dataset for demonstration
    df = pd.read_csv(DATASET_PATH, header=1)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    sample_subset = df.sample(5, random_state=42)
    true_labels = sample_subset['class'].values
    sample_features = sample_subset.drop(columns=['class'])
    
    preds, probs = predict_sample(sample_features, MODEL_SAVE_PATH)
    
    print(f"{'Sample #':<10} | {'True Status':<15} | {'Predicted Status':<20} | {'PD Confidence':<15}")
    print("-" * 70)
    for i in range(len(preds)):
        true_str = "PD Patient (1)" if true_labels[i] == 1 else "Healthy (0)"
        pred_str = "PD Patient (1)" if preds[i] == 1 else "Healthy (0)"
        conf_str = f"{probs[i]*100:.2f}%" if probs is not None else "N/A"
        print(f"{i+1:<10} | {true_str:<15} | {pred_str:<20} | {conf_str:<15}")

if __name__ == "__main__":
    main()
