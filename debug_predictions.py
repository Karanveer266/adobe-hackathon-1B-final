import pandas as pd
import numpy as np
from main import PDFHeadingDetectionSystem
from config import Config
from collections import Counter

def debug_pipeline():
    print("🔍 DEBUGGING PIPELINE INTEGRATION")
    print("="*50)
    
    # Load system
    system = PDFHeadingDetectionSystem()
    
    # Load a small subset of synthetic data
    X = pd.read_csv(Config.FEATURES_FILE)
    y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()
    
    # Take first 1000 samples for debugging
    X_debug = X.head(1000).values
    y_debug = y[:1000]
    
    print(f"Debugging with {len(y_debug)} samples")
    print("True label distribution:", Counter(y_debug))
    
    # Create mock elements
    mock_elements = [{'text': f'debug_sample_{i}', 'font_size': 12} for i in range(len(X_debug))]
    
    # Get predictions from complete pipeline
    predictions = system.classification_system.predict(X_debug, mock_elements)
    
    print(f"\nPipeline predictions: {len(predictions)}")
    print(f"Prediction rate: {len(predictions)/len(y_debug):.1%}")
    
    if predictions:
        pred_types = [p['type'] for p in predictions]
        pred_confidences = [p['confidence'] for p in predictions]
        
        print("Predicted label distribution:", Counter(pred_types))
        print(f"Average confidence: {np.mean(pred_confidences):.3f}")
        print(f"Confidence range: {np.min(pred_confidences):.3f} - {np.max(pred_confidences):.3f}")
        
        # Check confidence thresholds
        binary_conf = [p.get('binary_confidence', 0) for p in predictions]
        hierarchical_conf = [p.get('hierarchical_confidence', 0) for p in predictions]
        
        print(f"Binary confidence avg: {np.mean(binary_conf):.3f}")
        print(f"Hierarchical confidence avg: {np.mean(hierarchical_conf):.3f}")
        
        # Show some example predictions
        print("\nExample predictions:")
        for i, pred in enumerate(predictions[:10]):
            idx = pred['element_index']
            true_label = y_debug[idx] if idx < len(y_debug) else "unknown"
            print(f"  Sample {idx}: True={true_label}, Pred={pred['type']}, Conf={pred['confidence']:.3f}")

if __name__ == "__main__":
    debug_pipeline()