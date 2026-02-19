"""
Evaluation Script for Phishing Classifier
"""
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from models.phishing_classifier import PhishingClassifier
from training.dataset import PhishingDataset
from app.config import MODELS_DIR, USE_GPU

logger = logging.getLogger(__name__)


def evaluate_model(model_path: str = None, test_urls: list = None, test_labels: list = None):
    """
    Evaluate model performance

    Args:
        model_path: Path to trained model
        test_urls: Test URLs
        test_labels: Test labels (0=legitimate, 1=phishing)
    """
    # Set device
    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    model_path = model_path or (MODELS_DIR / "phishing_classifier.pt")
    model = PhishingClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logger.info(f"Loaded model from {model_path}")

    # Load test data
    if test_urls is None or test_labels is None:
        dataset = PhishingDataset()
        dataset.load_from_csv()
        _, _, test_urls, _, _, test_labels = dataset.split_train_val_test()

    # Create test dataset and loader
    test_dataset = PhishingDataset(test_urls, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluation
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features).squeeze()
            probabilities = outputs.cpu().numpy()
            predictions = (probabilities >= 0.5).astype(int)

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_probabilities)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_roc:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    # False positive rate and false negative rate
    fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    fnr = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0

    print(f"\nError Rates:")
    print(f"  False Positive Rate:  {fpr:.4f}")
    print(f"  False Negative Rate:  {fnr:.4f}")

    print("\n" + "="*50)

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Legitimate', 'Phishing']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm.tolist()
    }


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run evaluation
    evaluate_model()
