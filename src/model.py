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

def build_tuned_adaboost(
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 4,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features = 'sqrt',
    random_state: int = 42
) -> AdaBoostClassifier:
    """Tuned AdaBoost with regularized trees for superior bias-variance tradeoff."""
    base = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state
    )
    return AdaBoostClassifier(
        estimator=base,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )

def build_rbf_svm(C: float = 5.0, gamma: str = 'scale', random_state: int = 42):
    """Constructs RBF Support Vector Classifier with probability calibration."""
    from sklearn.svm import SVC
    return SVC(C=C, kernel='rbf', gamma=gamma, probability=True, random_state=random_state)

def build_mlp(hidden_layer_sizes=(128, 64), max_iter: int = 500, alpha: float = 0.001, random_state: int = 42):
    """Constructs Multi-Layer Perceptron (Dense Neural Network) for tabular features."""
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=True,
        random_state=random_state
    )

def build_xgboost(n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.05, random_state: int = 42):
    """Constructs XGBoost Classifier for boosting family comparison."""
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=random_state
    )

def build_voting_ensemble(models_dict: dict, voting: str = 'soft'):
    """Constructs VotingClassifier combining complementary models."""
    from sklearn.ensemble import VotingClassifier
    estimators = list(models_dict.items())
    return VotingClassifier(estimators=estimators, voting=voting)

def build_stacking_classifier(base_models_dict: dict, cv: int = 5):
    """
    Constructs StackingClassifier using Logistic Regression meta-learner
    trained on Out-Of-Fold (OOF) cross-validated probability outputs.
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    estimators = list(base_models_dict.items())
    meta_learner = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=cv,
        n_jobs=-1
    )


def save_trained_pipeline(pipeline_dict: dict, file_path: str):
    """Saves model, scaler, pca, and metadata to disk."""
    joblib.dump(pipeline_dict, file_path)
    print(f"[Model] Saved trained model pipeline to {file_path}")

def load_trained_pipeline(file_path: str) -> dict:
    """Loads saved model pipeline from disk."""
    pipeline_dict = joblib.load(file_path)
    print(f"[Model] Successfully loaded pipeline from {file_path}")
    return pipeline_dict
