"""
Phishing Classifier - PyTorch Model for URL Classification
"""
import logging
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from app.config import (
    MODEL_PATH, ONNX_MODEL_PATH, MODEL_TYPE,
    USE_GPU, URL_FEATURE_DIM,
    RISK_THRESHOLDS, PHISHING_THRESHOLD
)
from app.utils.url_processor import URLProcessor
from app.utils.content_fetcher import ContentFetcher

logger = logging.getLogger(__name__)


class URLDataset(Dataset):
    """Dataset for URL classification"""

    def __init__(self, urls: list, labels: list = None):
        self.urls = urls
        self.labels = labels
        self.processor = URLProcessor()

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        features = self.processor.extract_features(url)
        return features


class PhishingClassifier(nn.Module):
    """
    URL-Only Phishing URL Classifier

    Architecture (must match trained model):
    - Input: 20 URL lexical features
    - Hidden: 64 -> 32 -> 16
    - Output: 1 (phishing probability)

    Features:
    - URL structure (length, dots, dashes, underscores)
    - Domain analysis (TLD, IP address, subdomains)
    - Protocol (HTTPS, port)
    - Path analysis (length, depth, query params)
    - Keyword detection (brand names, sensitive words)
    """

    def __init__(self, input_dim: int = 20):
        super(PhishingClassifier, self).__init__()

        self.input_dim = input_dim

        # Main network - exact architecture from training
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Tensor of shape (batch_size, input_dim) with URL features

        Returns:
            Tensor of shape (batch_size, 1) with phishing probabilities
        """
        return self.net(x)


class PhishingClassifierModel:
    """
    Wrapper class for PhishingClassifier with inference logic
    """

    def __init__(self):
        self.device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = URLProcessor()
        self.content_fetcher = None
        self._load_model()

    def _load_model(self):
        """Load or initialize the model"""
        try:
            if MODEL_TYPE == 'onnx' and ONNX_MODEL_PATH.exists():
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH))
                self.model_type = 'onnx'
                logger.info(f"Loaded ONNX model from {ONNX_MODEL_PATH}")
            elif MODEL_PATH.exists():
                self.model = PhishingClassifier()
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.model_type = 'pytorch'
                logger.info(f"Loaded PyTorch model from {MODEL_PATH}")
            else:
                # Initialize new model (for training)
                self.model = PhishingClassifier()
                self.model.to(self.device)
                self.model.eval()
                self.model_type = 'pytorch'
                logger.warning("No trained model found, using initialized model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to initialized model
            self.model = PhishingClassifier()
            self.model.to(self.device)
            self.model.eval()
            self.model_type = 'pytorch'

    def predict(self, url: str, fetch_content: bool = False) -> Dict[str, Any]:
        """
        Predict if a URL is phishing

        Args:
            url: URL to classify
            fetch_content: Whether to fetch and analyze page content

        Returns:
            Dictionary with prediction results:
            {
                'is_phishing': bool,
                'confidence': float (0-1),
                'risk_level': str,
                'features': dict,
                'processing_time_ms': int
            }
        """
        import time

        start_time = time.time()

        try:
            # Extract URL features
            url_features = self.processor.extract_features(url)

            if 'error' in url_features:
                return {
                    'is_phishing': False,
                    'confidence': 0.0,
                    'risk_level': 'LOW',
                    'error': url_features['error']
                }

            # Prepare feature vector for model
            feature_vector = self._prepare_feature_vector(url_features)

            # Run inference
            if self.model_type == 'onnx':
                phishing_prob = self._predict_onnx(feature_vector)
            else:
                phishing_prob = self._predict_pytorch(feature_vector)

            # Determine classification
            is_phishing = phishing_prob >= PHISHING_THRESHOLD

            # Calculate risk level
            risk_level = self._get_risk_level(phishing_prob)

            # Add content features if requested
            if fetch_content:
                if self.content_fetcher is None:
                    self.content_fetcher = ContentFetcher()
                content_features = self.content_fetcher.fetch_content(url)
                url_features.update(content_features)

            processing_time = int((time.time() - start_time) * 1000)

            result = {
                'is_phishing': is_phishing,
                'confidence': float(phishing_prob),
                'risk_level': risk_level,
                'features': url_features,
                'processing_time_ms': processing_time
            }

            logger.info(f"Prediction: {url} -> is_phishing={is_phishing}, "
                       f"confidence={phishing_prob:.3f}, risk={risk_level}")

            return result

        except Exception as e:
            logger.error(f"Prediction failed for {url}: {e}", exc_info=True)
            return {
                'is_phishing': False,
                'confidence': 0.0,
                'risk_level': 'LOW',
                'error': str(e)
            }

    def _prepare_feature_vector(self, features: Dict[str, Any]) -> torch.Tensor:
        """
        Convert feature dict to 20-dimensional tensor matching trained model

        Feature order (must match training):
        1. URL长度
        2. 域名长度
        3. 点号数量
        4. 是否有@符号
        5. 连字符数量
        6. 下划线数量
        7. 是否为IP地址
        8. 子域名层级
        9. 域名是否有连字符
        10. 域名是否有下划线
        11. 免费TLD
        12. .xyz TLD
        13. 常见TLD
        14. 是否HTTPS
        15. 非标准端口
        16. 路径长度
        17. 路径深度
        18. 是否有查询参数
        19. 敏感词
        20. 品牌词
        """
        import re
        from urllib.parse import urlparse

        # Get URL from features dict (try different keys)
        url = features.get('original_url') or features.get('url', '')

        try:
            parsed = urlparse(url)
        except:
            parsed = urlparse('http://invalid')

        url_lower = url.lower()
        hostname = (parsed.hostname or '').lower()

        vector = np.array([
            len(url),                                    # 1. URL长度
            len(hostname),                                # 2. 域名长度
            url.count('.'),                               # 3. 点号数量
            1 if '@' in url else 0,                     # 4. 是否有@符号
            url.count('-'),                               # 5. 连字符数量
            url.count('_'),                               # 6. 下划线数量
            1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname) else 0,  # 7. 是否为IP地址
            max(0, hostname.count('.') - 2),              # 8. 子域名层级
            1 if hostname.count('-') > 0 else 0,         # 9. 域名是否有连字符
            1 if hostname.count('_') > 0 else 0,         # 10. 域名是否有下划线
            1 if hostname.endswith(('.tk', '.ml', '.ga', '.cf', '.gq')) else 0,  # 11. 免费TLD
            1 if hostname.endswith('.xyz') else 0,        # 12. .xyz TLD
            1 if any(hostname.endswith(tld) for tld in ['.com', '.org', '.net', '.edu']) else 0,  # 13. 常见TLD
            1 if parsed.scheme == 'https' else 0,         # 14. 是否HTTPS
            1 if parsed.port and parsed.port not in [80, 443] else 0,  # 15. 非标准端口
            len(parsed.path) if parsed.path else 0,        # 16. 路径长度
            (parsed.path or '').count('/'),                # 17. 路径深度
            1 if parsed.query else 0,                     # 18. 是否有查询参数
            1 if any(word in url_lower for word in ['login', 'signin', 'account', 'verify', 'secure']) else 0,  # 19. 敏感词
            1 if any(word in url_lower for word in ['apple', 'google', 'paypal', 'amazon', 'facebook', 'microsoft']) else 0,  # 20. 品牌词
        ], dtype=np.float32)

        return torch.from_numpy(vector).unsqueeze(0).to(self.device)

    def _predict_pytorch(self, feature_vector: torch.Tensor) -> float:
        """Run prediction using PyTorch model"""
        with torch.no_grad():
            output = self.model(feature_vector)
            return output.cpu().item()

    def _predict_onnx(self, feature_vector: np.ndarray) -> float:
        """Run prediction using ONNX model"""
        inputs = {self.ort_session.get_inputs()[0].name: feature_vector.cpu().numpy()}
        outputs = self.ort_session.run(None, inputs)
        return float(outputs[0][0][0])

    def _get_risk_level(self, confidence: float) -> str:
        """Convert confidence to risk level"""
        if confidence >= RISK_THRESHOLDS['CRITICAL']:
            return 'CRITICAL'
        elif confidence >= RISK_THRESHOLDS['HIGH']:
            return 'HIGH'
        elif confidence >= RISK_THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'

    def save_model(self, path: str = None):
        """Save the model"""
        save_path = path or MODEL_PATH
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    def export_to_onnx(self, path: str = None):
        """Export model to ONNX format"""
        save_path = path or ONNX_MODEL_PATH

        # Create dummy input with correct dimension (20 features)
        dummy_input = torch.randn(1, 20).to(self.device)

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,
            input_names=['url_features'],
            output_names=['phishing_probability'],
            dynamic_axes={
                'url_features': {0: 'batch_size'},
                'phishing_probability': {0: 'batch_size'}
            }
        )

        logger.info(f"Model exported to ONNX: {save_path}")


# Singleton instance
_classifier = None


def get_classifier() -> PhishingClassifierModel:
    """Get or create classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = PhishingClassifierModel()
    return _classifier
