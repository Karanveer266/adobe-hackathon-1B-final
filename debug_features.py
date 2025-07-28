import pandas as pd
from config import Config

def debug_feature_consistency():
    print("🔍 DEBUGGING FEATURE CONSISTENCY")
    print("="*50)
    
    # Load synthetic data
    X = pd.read_csv(Config.FEATURES_FILE)
    y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()
    
    print(f"Loaded data shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Check for any NaN or infinite values
    nan_cols = X.columns[X.isnull().any()].tolist()
    inf_cols = X.columns[np.isinf(X).any()].tolist()
    
    print(f"Columns with NaN: {nan_cols}")
    print(f"Columns with Inf: {inf_cols}")
    
    # Check label distribution
    from collections import Counter
    label_dist = Counter(y)
    print(f"Label distribution: {label_dist}")
    
    # Check feature ranges
    print("\nFeature statistics:")
    print(X.describe())

if __name__ == "__main__":
    import numpy as np
    debug_feature_consistency()