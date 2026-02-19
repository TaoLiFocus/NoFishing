"""
Training Script for Phishing Classifier
"""
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from models.phishing_classifier import PhishingClassifier
from app.utils.url_processor import URLProcessor
from app.config import MODELS_DIR, USE_GPU

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhishingDataset(Dataset):
    """Dataset for phishing URL classification"""

    def __init__(self, urls: list, labels: list):
        self.urls = urls
        self.labels = labels
        self.processor = URLProcessor()

        # Pre-extract features for all URLs
        logger.info("Extracting features for dataset...")
        self.features = []
        for url in urls:
            feats = self.processor.extract_features(url)
            self.features.append(feats)
        logger.info(f"Extracted features for {len(urls)} URLs")

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]
        features = self.features[idx]

        # Convert to feature vector
        feature_vector = self._features_to_vector(features)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return feature_vector, label_tensor

    def _features_to_vector(self, features: dict) -> torch.Tensor:
        """Convert feature dict to tensor vector"""
        feature_indices = {
            'url_length': 0,
            'domain_length': 1,
            'subdomain_count': 2,
            'path_length': 3,
            'path_depth': 4,
            'query_param_count': 5,
            'dot_count': 6,
            'digit_count': 7,
            'special_char_count': 8,
            'percent_encode_count': 9,
            'has_ip_address': 10,
            'has_https': 11,
            'has_dash_in_domain': 12,
            'has_suspicious_tld': 13,
            'has_suspicious_keyword': 14,
            'has_brand_name': 15,
            'has_at_symbol': 16,
            'has_non_ascii': 17,
            'has_freenom_domain': 18,
            'has_numeric_domain': 19,
            'subdomain_length_ratio': 20,
        }

        vector = np.zeros(50, dtype=np.float32)

        for name, idx in feature_indices.items():
            value = features.get(name, 0)
            if isinstance(value, bool):
                vector[idx] = 1.0 if value else 0.0
            else:
                vector[idx] = float(value) if value is not None else 0.0

        if 'heuristic_score' in features:
            vector[21] = features['heuristic_score']

        return torch.from_numpy(vector)


class SyntheticDataGenerator:
    """Generate synthetic training data for demonstration"""

    # Phishing URL patterns
    PHISHING_PATTERNS = [
        "http://{subdomain}.verify-login.{tld}/auth",
        "http://{domain}.update-account.{tld}/confirm",
        "http://{ip}/secure-login",
        "http://{subdomain}.@{domain}.{tld}/signin",
        "http://www.{brand}-{variation}.{tld}/login",
    ]

    # Legitimate URL patterns
    LEGITIMATE_PATTERNS = [
        "https://www.{domain}.com/{path}",
        "https://{subdomain}.{domain}.com/{path}",
        "https://www.{domain}.com",
        "https://api.{domain}.com/v1/endpoint",
    ]

    PHISHING_TLDS = ['.xyz', '.top', '.tk', '.ml', '.ga']
    LEGITIMATE_TLDS = ['.com', '.org', '.net', '.io', '.edu']

    BRANDS = ['apple', 'google', 'microsoft', 'amazon', 'paypal']
    VARIATIONS = ['secure', 'verify', 'login', 'account', 'update']

    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples

    def generate(self) -> tuple[list[str], list[int]]:
        """Generate synthetic training data"""
        urls = []
        labels = []

        # Generate phishing URLs
        for _ in range(self.num_samples // 2):
            pattern = np.random.choice(self.PHISHING_PATTERNS)
            url = self._fill_pattern(pattern, is_phishing=True)
            urls.append(url)
            labels.append(1)

        # Generate legitimate URLs
        for _ in range(self.num_samples // 2):
            pattern = np.random.choice(self.LEGITIMATE_PATTERNS)
            url = self._fill_pattern(pattern, is_phishing=False)
            urls.append(url)
            labels.append(0)

        # Shuffle
        combined = list(zip(urls, labels))
        np.random.shuffle(combined)
        urls, labels = zip(*combined)

        return list(urls), list(labels)

    def _fill_pattern(self, pattern: str, is_phishing: bool) -> str:
        """Fill pattern with random values"""
        import random
        import string

        tlds = self.PHISHING_TLDS if is_phishing else self.LEGITIMATE_TLDS
        tld = np.random.choice(tlds)

        domains = ['example', 'service', 'company', 'website']
        subdomains = ['api', 'mail', 'portal', 'secure', 'auth']

        result = pattern

        if '{tld}' in result:
            result = result.replace('{tld}', tld)
        if '{domain}' in result:
            result = result.replace('{domain}', np.random.choice(domains))
        if '{subdomain}' in result:
            result = result.replace('{subdomain}', np.random.choice(subdomains))
        if '{ip}' in result:
            result = result.replace('{ip}', f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}")
        if '{brand}' in result:
            result = result.replace('{brand}', np.random.choice(self.BRANDS))
        if '{variation}' in result:
            result = result.replace('{variation}', np.random.choice(self.VARIATIONS))
        if '{path}' in result:
            path = ''.join(random.choices(string.ascii_lowercase + '/', k=np.random.randint(5, 30)))
            result = result.replace('{path}', path)

        return result


def train_model(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_split: float = 0.8
):
    """
    Train the phishing classifier model

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        train_split: Fraction of data to use for training
    """
    # Set device
    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Generate synthetic data
    logger.info("Generating synthetic training data...")
    generator = SyntheticDataGenerator(num_samples=1000)
    urls, labels = generator.generate()
    logger.info(f"Generated {len(urls)} samples")

    # Create dataset
    dataset = PhishingDataset(urls, labels)

    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = PhishingClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            model_path = MODELS_DIR / "phishing_classifier.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        # Early stopping
        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info("Training complete!")


if __name__ == '__main__':
    train_model(
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
