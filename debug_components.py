import pandas as pd
import numpy as np
from classifiers import create_classification_system
from config import Config
from sklearn.model_selection import train_test_split

def debug_individual_components():
    print("🔍 DEBUGGING INDIVIDUAL COMPONENTS")
    print("="*50)
    
    # Load classifier
    classifier = create_classification_system()
    if not classifier.load_models():
        print("❌ Could not load models")
        return
    
    # Load data
    X = pd.read_csv(Config.FEATURES_FILE)
    y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()
    
    # Use same split as training
    _, X_test, _, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Test set size: {X_test.shape}")
    
    # Test binary classifier directly
    print("\n1. TESTING BINARY CLASSIFIER:")
    y_binary_test = ['heading' if label in Config.HEADING_TYPES else 'non-heading' for label in y_test]
    
    # Process features
    X_processed = classifier._preprocess_prediction_features(X_test)
    
    binary_predictions = classifier.binary_classifier.predict(X_processed)
    binary_accuracy = sum(1 for true, pred in zip(y_binary_test, binary_predictions) if true == pred) / len(y_binary_test)
    
    print(f"Binary accuracy: {binary_accuracy:.4f}")
    print("Binary prediction distribution:", {label: list(binary_predictions).count(label) for label in set(binary_predictions)})
    
    # Test hierarchical classifier on heading samples only
    print("\n2. TESTING HIERARCHICAL CLASSIFIER:")
    heading_indices = [i for i, pred in enumerate(binary_predictions) if pred == 'heading']
    
    if heading_indices:
        X_headings = X_processed[heading_indices]
        y_headings_true = [y_test[i] for i in heading_indices]
        
        hierarchical_predictions = classifier.hierarchical_classifier.predict(X_headings)
        hierarchical_accuracy = sum(1 for true, pred in zip(y_headings_true, hierarchical_predictions) if true == pred) / len(y_headings_true)
        
        print(f"Hierarchical accuracy on detected headings: {hierarchical_accuracy:.4f}")
        print("Hierarchical prediction distribution:", {label: list(hierarchical_predictions).count(label) for label in set(hierarchical_predictions)})
        
        # Show confidence distributions
        hierarchical_probs = classifier.hierarchical_classifier.predict_proba(X_headings)
        max_probs = np.max(hierarchical_probs, axis=1)
        print(f"Hierarchical confidence range: {np.min(max_probs):.3f} - {np.max(max_probs):.3f}")
        print(f"Above threshold ({Config.HIERARCHICAL_CONFIDENCE_THRESHOLD}): {sum(1 for p in max_probs if p >= Config.HIERARCHICAL_CONFIDENCE_THRESHOLD)}/{len(max_probs)}")
    
    print(f"\n3. CONFIDENCE THRESHOLDS:")
    print(f"Binary threshold: {Config.BINARY_CONFIDENCE_THRESHOLD}")
    print(f"Hierarchical threshold: {Config.HIERARCHICAL_CONFIDENCE_THRESHOLD}")

if __name__ == "__main__":
    debug_individual_components()