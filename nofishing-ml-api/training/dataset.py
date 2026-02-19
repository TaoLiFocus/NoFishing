"""
Dataset Module - Load and prepare training data
"""
import logging
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split

from app.utils.url_processor import URLProcessor

logger = logging.getLogger(__name__)


class PhishingDataset:
    """
    Load and manage phishing dataset

    Expected CSV format:
    url,label
    http://example.com,0
    http://phishing-site.com,1
    """

    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path) if data_path else Path("data/phishing_dataset.csv")
        self.processor = URLProcessor()
        self.data = None
        self.urls = None
        self.labels = None

    def load_from_csv(self, path: str = None) -> pd.DataFrame:
        """Load dataset from CSV file"""
        path = Path(path) if path else self.data_path

        if not path.exists():
            logger.error(f"Dataset file not found: {path}")
            return pd.DataFrame()

        try:
            self.data = pd.read_csv(path)
            logger.info(f"Loaded dataset from {path}: {len(self.data)} samples")

            # Validate columns
            if 'url' not in self.data.columns or 'label' not in self.data.columns:
                raise ValueError("CSV must contain 'url' and 'label' columns")

            self.urls = self.data['url'].tolist()
            self.labels = self.data['label'].tolist()

            return self.data

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return pd.DataFrame()

    def load_from_lists(self, urls: List[str], labels: List[int]) -> None:
        """Load dataset from lists"""
        self.urls = urls
        self.labels = labels
        self.data = pd.DataFrame({'url': urls, 'label': labels})
        logger.info(f"Loaded dataset from lists: {len(urls)} samples")

    def split_train_val_test(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[List, List, List, List, List, List]:
        """
        Split dataset into train, validation, and test sets

        Returns:
            (train_urls, val_urls, test_urls, train_labels, val_labels, test_labels)
        """
        if self.urls is None or self.labels is None:
            raise ValueError("No data loaded. Call load_from_csv() first.")

        test_ratio = 1 - train_ratio - val_ratio

        # First split: train + val vs test
        train_val_urls, test_urls, train_val_labels, test_labels = train_test_split(
            self.urls, self.labels,
            test_size=test_ratio,
            random_state=42,
            stratify=self.labels
        )

        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_urls, val_urls, train_labels, val_labels = train_test_split(
            train_val_urls, train_val_labels,
            test_size=val_ratio_adjusted,
            random_state=42,
            stratify=train_val_labels
        )

        logger.info(
            f"Dataset split: train={len(train_urls)}, "
            f"val={len(val_urls)}, test={len(test_urls)}"
        )

        return (
            train_urls, val_urls, test_urls,
            train_labels, val_labels, test_labels
        )

    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        if self.data is None:
            return {}

        label_counts = self.data['label'].value_counts()
        total = len(self.data)

        return {
            'total_samples': total,
            'phishing_samples': int(label_counts.get(1, 0)),
            'legitimate_samples': int(label_counts.get(0, 0)),
            'phishing_ratio': float(label_counts.get(1, 0) / total),
            'legitimate_ratio': float(label_counts.get(0, 0) / total),
        }


def create_sample_dataset(output_path: str = "data/phishing_dataset.csv"):
    """Create a sample dataset for testing"""
    sample_data = {
        'url': [
            # Legitimate URLs
            'https://www.google.com',
            'https://www.facebook.com',
            'https://www.amazon.com',
            'https://www.microsoft.com',
            'https://www.apple.com',
            'https://github.com',
            'https://stackoverflow.com',
            'https://www.wikipedia.org',
            # Phishing URLs
            'http://verify-login.apple.com.tk',
            'http://192.168.1.1/secure-login',
            'http://@apple.com.icu/update-account',
            'http://secure-paypla.xyz/login',
            'http://apple-verify-account.top/confirm',
            'http://free-apple-gift.ml/claim',
            'http://account-verify-ga.ml/signin',
        ],
        'label': [
            0, 0, 0, 0, 0, 0, 0, 0,  # 8 legitimate
            1, 1, 1, 1, 1, 1, 1,      # 7 phishing
        ]
    }

    df = pd.DataFrame(sample_data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Created sample dataset: {output_path}")

    return df


if __name__ == '__main__':
    # Create sample dataset
    create_sample_dataset()
