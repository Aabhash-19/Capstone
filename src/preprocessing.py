import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Applies SMOTE to balance the dataset minority class.
    Original: 564 positive (PD), 192 negative (Healthy) -> Balanced: 564 positive, 564 negative.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[Preprocessing] SMOTE Applied. Original: {y.value_counts().to_dict()} -> Balanced: {y_res.value_counts().to_dict()}")
    return X_res, y_res

def prepare_pipeline_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, n_components: int = 6, random_state: int = 42):
    """
    Executes the exact paper preprocessing workflow:
    1. SMOTE class balancing
    2. StandardScaler normalization
    3. PCA feature extraction (retaining n_components=6)
    4. Train-Test 80:20 split
    """
    # 1. SMOTE Balancing
    X_res, y_res = apply_smote(X, y, random_state=random_state)
    
    # 2. Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # 3. PCA Dimensionality Reduction (6 components)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"[Preprocessing] PCA Retained {n_components} Principal Components.")
    print(f"[Preprocessing] Total Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_res, test_size=test_size, random_state=random_state
    )
    
    print(f"[Preprocessing] Train-Test Split ({int((1-test_size)*100)}:{int(test_size*100)}) -> Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, scaler, pca
