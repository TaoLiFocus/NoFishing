# -*- coding: utf-8 -*-
"""
Debug Model Loading
调试模型加载过程
"""
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '.')

from app.config import MODEL_PATH

print("=" * 70)
print("Debug Model Loading")
print("=" * 70)

# ============================================================================
# Test 1: Load model using PhishingClassifierModel wrapper
# ============================================================================
print("\n[Test 1] Using PhishingClassifierModel wrapper")

from models.phishing_classifier import PhishingClassifierModel

wrapper = PhishingClassifierModel()
print(f"  Model type: {wrapper.model_type}")
print(f"  Model class: {type(wrapper.model).__name__}")

# Check a prediction
url = "http://apple-verify.tk/account"
result = wrapper.predict(url)
print(f"  Prediction for {url}:")
print(f"    is_phishing: {result['is_phishing']}")
print(f"    confidence: {result['confidence']:.4f}")

# ============================================================================
# Test 2: Load model directly
# ============================================================================
print("\n[Test 2] Loading model directly")

class URLClassifier(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
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
        return self.net(x)

model = URLClassifier(input_dim=20)
state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(state_dict)
model.eval()

print(f"  Model class: {type(model).__name__}")

# Check weights
print("\n[Test 3] Check model weights")
print(f"  net.0 weight mean: {model.net[0].weight.mean().item():.6f}")
print(f"  net.0 weight std: {model.net[0].weight.std().item():.6f}")
print(f"  net.10 weight shape: {model.net[10].weight.shape}")

# ============================================================================
# Test 4: Compare wrapper model weights
# ============================================================================
print("\n[Test 4] Wrapper model weights")
print(f"  net.0 weight mean: {wrapper.model.net[0].weight.mean().item():.6f}")
print(f"  net.0 weight std: {wrapper.model.net[0].weight.std().item():.6f}")
print(f"  net.10 weight shape: {wrapper.model.net[10].weight.shape}")

# ============================================================================
# Test 5: Manual feature extraction and prediction
# ============================================================================
print("\n[Test 5] Manual prediction with direct model")

import re
from urllib.parse import urlparse

def extract_features(url):
    parsed = urlparse(url)
    url_lower = url.lower()
    hostname = (parsed.hostname or '').lower()

    return np.array([
        len(url), len(hostname), url.count('.'), 1 if '@' in url else 0,
        url.count('-'), url.count('_'),
        1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname) else 0,
        max(0, hostname.count('.') - 2),
        1 if hostname.count('-') > 0 else 0, 1 if hostname.count('_') > 0 else 0,
        1 if hostname.endswith(('.tk', '.ml', '.ga', '.cf', '.gq')) else 0,
        1 if hostname.endswith('.xyz') else 0,
        1 if any(hostname.endswith(tld) for tld in ['.com', '.org', '.net', '.edu']) else 0,
        1 if parsed.scheme == 'https' else 0,
        1 if parsed.port and parsed.port not in [80, 443] else 0,
        len(parsed.path) if parsed.path else 0,
        (parsed.path or '').count('/'),
        1 if parsed.query else 0,
        1 if any(w in url_lower for w in ['login', 'signin', 'account', 'verify', 'secure']) else 0,
        1 if any(w in url_lower for w in ['apple', 'google', 'paypal', 'amazon', 'facebook', 'microsoft']) else 0,
    ], dtype=np.float32)

features = extract_features(url)
print(f"  Features shape: {features.shape}")
print(f"  Features: {features}")

with torch.no_grad():
    prob = model(torch.FloatTensor(features).unsqueeze(0)).item()

print(f"  Direct prediction: {prob:.4f}")

print("\n" + "=" * 70)
