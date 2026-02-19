# -*- coding: utf-8 -*-
"""
Test Flask ML API with Trained Model
测试 Flask ML API 与训练好的模型
"""
import sys
import os
import time

# Add nofishing-ml-api to path
API_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nofishing-ml-api')
sys.path.insert(0, API_DIR)

# Also add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("Testing Flask ML API with Trained Model")
print("=" * 70)

# ============================================================================
# 直接测试模型（不启动 Flask）
# ============================================================================
print("\n[Direct Model Test]")

from models.phishing_classifier import get_classifier
from app.config import MODEL_PATH

print(f"  Model path: {MODEL_PATH}")
print(f"  Model exists: {MODEL_PATH.exists()}")

classifier = get_classifier()
print(f"  Classifier loaded: {classifier is not None}")
print(f"  Model type: {classifier.model_type if hasattr(classifier, 'model_type') else 'unknown'}")

# ============================================================================
# 测试 URL 分类
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
# 性能测试
# ============================================================================
print("\n[Performance Test]")

# 测试100次推理取平均
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
# 完成
# ============================================================================
print("\n" + "=" * 70)
print("Test Complete!")
print("\nSummary:")
print(f"  Model: {MODEL_PATH}")
print(f"  Accuracy: {accuracy:.1f}%")
print(f"  Avg Response: {avg_time:.1f}ms")
print(f"  Status: {'PASS' if accuracy >= 90 and avg_time < 100 else 'FAIL'}")
print("=" * 70)
