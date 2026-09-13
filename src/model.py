import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def build_adaboost_model(
    n_estimators: int = 500,
    learning_rate: float = 1.0,
    max_depth: int = 7,
    random_state: int = 42
) -> AdaBoostClassifier:
    """
    Constructs AdaBoost ensemble classifier as described in Bukhari & Ogudo (2024).
    
    Parameters:
    -----------
    n_estimators : int
        Number of decision tree weak learners (default = 500).
    learning_rate : float
        Learning rate shrink factor (default = 1.0).
    max_depth : int
        Maximum depth of base DecisionTree estimators (default = 7).
    random_state : int
        Random seed for reproducibility.
    """
    base_learner = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    
    adaboost_model = AdaBoostClassifier(
        estimator=base_learner,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    print(f"[Model] Built AdaBoost Classifier (base_depth={max_depth}, n_estimators={n_estimators}, learning_rate={learning_rate})")
    return adaboost_model

def save_trained_pipeline(pipeline_dict: dict, file_path: str):
    """Saves model, scaler, pca, and metadata to disk."""
    joblib.dump(pipeline_dict, file_path)
    print(f"[Model] Saved trained model pipeline to {file_path}")

def load_trained_pipeline(file_path: str) -> dict:
    """Loads saved model pipeline from disk."""
    pipeline_dict = joblib.load(file_path)
    print(f"[Model] Successfully loaded pipeline from {file_path}")
    return pipeline_dict
