# -*- coding: utf-8 -*-
"""
Test Trained Model Directly
直接测试训练好的模型
"""
import sys
import os
import time

# Set Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Testing Trained Model")
print("=" * 70)

# ============================================================================
# Import model
# ============================================================================
print("\n[Loading Model]")

from models.phishing_classifier import get_classifier
from app.config import MODEL_PATH

print(f"  Model path: {MODEL_PATH}")
print(f"  Model exists: {MODEL_PATH.exists()}")

classifier = get_classifier()
print(f"  Classifier loaded: {classifier is not None}")
print(f"  Model type: {classifier.model_type if hasattr(classifier, 'model_type') else 'unknown'}")

# ============================================================================
# Test URLs
# ============================================================================
print("\n[Classification Test]")

test_urls = [
    ("https://www.google.com/search", 0, "Legitimate Google"),
    ("http://192.168.1.1/login", 1, "IP Address Login"),
    ("http://apple-verify.tk/account", 1, "Apple Phishing"),
    ("https://github.com/user/repo", 0, "Legitimate GitHub"),
    ("http://paypal-secure.ml/signin", 1, "PayPal Phishing"),
    ("https://www.amazon.com/dp/product", 0, "Legitimate Amazon"),
    ("http://google-secure.cf/login", 1, "Google Phishing"),
    ("https://stackoverflow.com/questions", 0, "Legitimate StackOverflow"),
]

print(f"\n  {'URL':<50} {'Expected':>10} {'Actual':>10} {'Conf':>10} {'Time':>10}")
print("  " + "-" * 95)

correct = 0
total = len(test_urls)

for url, expected, desc in test_urls:
    start = time.time()
    result = classifier.predict(url, fetch_content=False)
    elapsed = int((time.time() - start) * 1000)

    actual = 1 if result['is_phishing'] else 0
    match = "OK" if actual == expected else "WRONG"
    status = "PHISH" if result['is_phishing'] else "SAFE"
    conf = f"{result['confidence']*100:.0f}%"

    if actual == expected:
        correct += 1

    print(f"  {url:<50} {match:>10} {status:>10} {conf:>10} {elapsed:>10}ms - {desc}")

accuracy = (correct / total) * 100
print(f"\n  Accuracy: {accuracy:.1f}% ({correct}/{total})")

# ============================================================================
# Performance test
# ============================================================================
print("\n[Performance Test]")

import random

test_samples = [
    "https://www.google.com/search",
    "http://apple-verify.tk/login",
    "http://192.168.1.1/account",
    "https://github.com/user/repo",
    "http://paypal-secure.cf/signin",
]

times = []
for _ in range(100):
    url = random.choice(test_samples)
    start = time.time()
    classifier.predict(url, fetch_content=False)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)

avg_time = sum(times) / len(times)
max_time = max(times)
min_time = min(times)

print(f"  Samples: 100")
print(f"  Average: {avg_time:.1f}ms")
print(f"  Min: {min_time:.1f}ms")
print(f"  Max: {max_time:.1f}ms")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Test Complete!")
print("\nSummary:")
print(f"  Model: {MODEL_PATH}")
print(f"  Accuracy: {accuracy:.1f}%")
print(f"  Avg Response: {avg_time:.1f}ms")
print(f"  Status: {'PASS' if accuracy >= 90 and avg_time < 100 else 'FAIL'}")
print("=" * 70)
