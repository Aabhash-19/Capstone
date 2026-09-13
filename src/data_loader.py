import pandas as pd
import numpy as np

def load_parkinson_data(csv_path: str):
    """
    Load Parkinson's disease speech features dataset from CSV.
    
    Parameters:
    -----------
    csv_path : str
        Path to pd_speech_features.csv file.
        
    Returns:
    --------
    X : pd.DataFrame
        754 speech feature columns (excluding 'id' and 'class').
    y : pd.Series
        Binary classification target ('class', where 1=PD, 0=Healthy).
    df : pd.DataFrame
        Complete loaded dataframe.
    """
    # Load dataset with header=1 to parse feature column names correctly
    df = pd.read_csv(csv_path, header=1)
    
    # Drop identifier 'id' column if present
    if 'id' in df.columns:
        X = df.drop(columns=['id', 'class'])
    else:
        X = df.drop(columns=['class'])
        
    y = df['class']
    
    print(f"[DataLoader] Successfully loaded dataset from {csv_path}")
    print(f"[DataLoader] Total Instances: {len(df)}")
    print(f"[DataLoader] Total Feature Attributes: {X.shape[1]}")
    print(f"[DataLoader] Target Class Distribution: {y.value_counts().to_dict()} (1=PD, 0=Healthy)")
    
    return X, y, df

if __name__ == "__main__":
    X, y, df = load_parkinson_data('/Users/irray/Desktop/Projects/Capstone /pd_speech_features.csv')
