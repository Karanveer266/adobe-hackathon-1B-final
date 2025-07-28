import pandas as pd
import numpy as np
from classifiers import create_classification_system
from config import Config
from sklearn.model_selection import train_test_split

def diagnose_hierarchical_bias():
    print("🔍 DIAGNOSING HIERARCHICAL CLASSIFIER BIAS")
    print("="*50)
    
    classifier = create_classification_system()
    classifier.load_models()
    
    # Load data
    X = pd.read_csv(Config.FEATURES_FILE)
    y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()
    
    # Get heading samples only
    heading_indices = [i for i, label in enumerate(y) if label in Config.HEADING_TYPES]
    X_headings = X.iloc[heading_indices].values
    y_headings = [y[i] for i in heading_indices]
    
    # Preprocess features
    X_processed = classifier._preprocess_prediction_features(X_headings)
    
    # Get all predictions (no confidence filtering)
    raw_predictions = classifier.hierarchical_classifier.predict(X_processed)
    raw_probabilities = classifier.hierarchical_classifier.predict_proba(X_processed)
    
    # Analyze predictions
    from collections import Counter
    pred_distribution = Counter(raw_predictions)
    print("Raw hierarchical predictions (no confidence filter):")
    for class_name, count in pred_distribution.items():
        percentage = count / len(raw_predictions) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Analyze confidence by class
    class_names = classifier.hierarchical_classifier.classes_
    max_probs = np.max(raw_probabilities, axis=1)
    
    print(f"\nConfidence analysis:")
    print(f"  Mean confidence: {np.mean(max_probs):.3f}")
    print(f"  Min confidence: {np.min(max_probs):.3f}")
    print(f"  Max confidence: {np.max(max_probs):.3f}")
    
    # Check confidence by predicted class
    for class_name in class_names:
        class_mask = raw_predictions == class_name
        if np.any(class_mask):
            class_confidences = max_probs[class_mask]
            above_threshold = np.sum(class_confidences >= Config.HIERARCHICAL_CONFIDENCE_THRESHOLD)
            print(f"  {class_name}: {above_threshold}/{len(class_confidences)} above threshold (avg: {np.mean(class_confidences):.3f})")

if __name__ == "__main__":
    diagnose_hierarchical_bias()