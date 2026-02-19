"""
NoFishing ML API Configuration
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Flask Configuration
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_TTL = int(os.getenv("REDIS_TTL", 3600))  # 1 hour

# Model Configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "pytorch")  # pytorch or onnx
MODEL_PATH = MODELS_DIR / "phishing_classifier_phishtank.pt"
ONNX_MODEL_PATH = MODELS_DIR / "phishing_classifier.onnx"
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
URL_FEATURE_DIM = int(os.getenv("URL_FEATURE_DIM", 20))  # URL-only feature dimension

# BERT Configuration
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "bert-base-uncased")
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", 128))

# Feature Extraction
USE_CONTENT_ANALYSIS = os.getenv("USE_CONTENT_ANALYSIS", "True").lower() == "true"
CONTENT_FETCH_TIMEOUT = int(os.getenv("CONTENT_FETCH_TIMEOUT", 5))

# Classification Threshold
PHISHING_THRESHOLD = float(os.getenv("PHISHING_THRESHOLD", 0.5))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Risk Level Thresholds
RISK_THRESHOLDS = {
    "LOW": 0.0,
    "MEDIUM": 0.3,
    "HIGH": 0.6,
    "CRITICAL": 0.8
}
