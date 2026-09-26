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

def apply_feature_selection(X_train, y_train, X_test, k: int = 100):
    """
    Applies SelectKBest (ANOVA F-value) feature selection fitted strictly on training data.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    print(f"[Preprocessing] Feature Selection: Reduced {X_train.shape[1]} features to {k} best features.")
    return X_train_selected, X_test_selected, selector

def prepare_leakage_free_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    n_components = 50,
    k_features = None,
    smote_k: int = 5,
    random_state: int = 42
):
    """
    Strict, scientifically controlled preprocessing pipeline:
    1. Train/Test split is performed FIRST to isolate unseen test data.
    2. SMOTE is fitted ONLY on the training split.
    3. StandardScaler is fitted on train, applied to test.
    4. Optional SelectKBest feature selection fitted on train, applied to test.
    5. PCA is fitted on train, applied to test.
    """
    # 1. Stratified Train-Test split FIRST
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. SMOTE on Training split only
    smote = SMOTE(k_neighbors=smote_k, random_state=random_state)
    X_train_res, y_train = smote.fit_resample(X_train_raw, y_train_raw)
    
    # 3. StandardScaler fitted on training split only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 4. Optional Feature Selection
    selector = None
    if k_features is not None and k_features < X.shape[1]:
        X_train_scaled, X_test_scaled, selector = apply_feature_selection(
            X_train_scaled, y_train, X_test_scaled, k=k_features
        )
        
    # 5. PCA fitted on training split only
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)
    
    var_ratio = np.sum(pca.explained_variance_ratio_)
    print(f"[Preprocessing] Leakage-Free Pipeline Configured:")
    print(f"  - Train samples: {X_train.shape[0]} (Balanced {y_train.value_counts().to_dict()})")
    print(f"  - Test samples: {X_test.shape[0]} (Untouched {y_test.value_counts().to_dict()})")
    print(f"  - PCA retained components: {X_train.shape[1]} (Explained Variance: {var_ratio:.4f})")
    
    return X_train, X_test, y_train, y_test, scaler, pca, selector

