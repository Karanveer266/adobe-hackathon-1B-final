# FILE: evaluate_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging

from classifiers import create_classification_system
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def evaluate_saved_models():
    """
    Loads pre-trained models and evaluates them on the test set.
    """
    print("🚀 Starting model evaluation...")

    # 1. Load the classification system and its trained models
    classifier = create_classification_system()
    if not classifier.load_models():
        print("❌ Error: Could not load models. Make sure you have trained them first.")
        return

    print("✅ Models loaded successfully.")

    # 2. Load the dataset that the models were trained on
    try:
        X = pd.read_csv(Config.FEATURES_FILE)
        y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()
    except FileNotFoundError:
        print("❌ Error: Training data not found. Please generate it by running the training script.")
        return

    # 3. Recreate the *exact* same test split to evaluate on unseen data
    # The random_state=42 ensures the split is identical to the one during training.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # The classifier's internal pre-processing (scaling/selection) will be applied during predict()
    X_test_values = X_test.values

    print(f"📊 Evaluating on test set with {len(y_test)} samples.")

    # 4. Create mock elements required by the predict function
    mock_elements = [{'text': f'test_{i}'} for i in range(len(X_test_values))]

    # 5. Get predictions from the complete system
    # This uses the standard predict method, which now works because the models are loaded.
    predictions = classifier.predict(X_test_values, mock_elements)
    
    # 6. Format predictions for evaluation
    predicted_labels = ['non-heading'] * len(y_test)
    for pred in predictions:
        # The element_index corresponds to the index in the UNSPLIT features array.
        # We need to map it back to the original index from X_test.
        original_index = X_test.index[pred['element_index']]
        # This is getting complicated. A simpler way is to just predict on the whole test set.
        # Let's simplify the logic to be more robust.
        pass # We will replace this section.

    # --- SIMPLIFIED AND CORRECTED LOGIC ---
    predicted_labels = ['non-heading'] * len(y_test)
    # The `predict` method will process all elements passed to it.
    # The returned predictions contain `element_index` which is the index *within that batch*.
    for pred in predictions:
        idx = pred['element_index']
        if 0 <= idx < len(predicted_labels):
            predicted_labels[idx] = pred['type']


    # 7. Print the final results
    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels)

    print("\n" + "="*50)
    print("          COMPLETE SYSTEM EVALUATION REPORT")
    print("="*50 + "\n")
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)
    print("="*50)


if __name__ == "__main__":
    evaluate_saved_models()