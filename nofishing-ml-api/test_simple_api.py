# -*- coding: utf-8 -*-
"""
Test Simplified ML API
测试简化后的ML API（纯URL词法特征）
"""
import sys
import os
import time

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Testing Simplified NoFishing ML API")
print("=" * 70)

# Import classifier
from models.url_classifier import get_classifier

print("\n[Initializing] Loading URL classifier...")
classifier = get_classifier()
print("  Classifier ready!")

# ============================================================================
# Test URLs
# ============================================================================
print("\n[Testing] URL classification...")

test_cases = [
    # (url, expected_is_phishing, description)
    ("https://www.google.com/search", False, "Legitimate Google"),
    ("http://192.168.1.1/login", True, "IP address login"),
    ("http://apple-verify.tk/account", True, "Apple phishing with free TLD"),
    ("https://github.com/user/repo", False, "Legitimate GitHub"),
    ("http://paypal-secure.ml/signin", True, "PayPal phishing with free TLD"),
    ("https://www.amazon.com/dp/product", False, "Legitimate Amazon"),
    ("http://google-secure.cf/login", True, "Google phishing with free TLD"),
    ("https://stackoverflow.com/questions", False, "Legitimate StackOverflow"),
    ("http://verify-apple.tk/account", True, "Apple brand impersonation"),
    ("https://www.facebook.com/about", False, "Legitimate Facebook"),
    ("http://appleid.com.tk/login", True, "Apple ID phishing"),
    ("https://www.microsoft.com/en-us/", False, "Legitimate Microsoft"),
    ("http://secure-netflix.ga/signin", True, "Netflix phishing"),
]

print(f"\n  {'URL':<50} {'Expected':>10} {'Actual':>10} {'Prob':>10} {'Risk':>10} {'Time':>8}")
print("  " + "-" * 105)

correct = 0
total = len(test_cases)

for url, expected, desc in test_cases:
    start = time.time()
    result = classifier.classify(url)
    elapsed = int((time.time() - start) * 1000)

    actual = result['is_phishing']
    match = "OK" if actual == expected else "WRONG"
    prob = result['probability']
    risk = result['risk_level']

    if actual == expected:
        correct += 1

    status_mark = "!" if actual else "."
    print(f"  {url:<50} {match:>10} {status_mark:>10} {prob*100:>10.0f}% {risk:>10} {elapsed:>8}ms - {desc}")

accuracy = (correct / total) * 100

# ============================================================================
# Performance test
# ============================================================================
print("\n[Performance] Testing response time...")

sample_urls = [
    "https://www.google.com/search",
    "http://apple-verify.tk/account",
    "https://github.com/user/repo",
    "http://paypal-secure.ml/signin",
]

times = []
for _ in range(100):
    url = sample_urls[len(times) % len(sample_urls)]
    start = time.time()
    classifier.classify(url)
    times.append((time.time() - start) * 1000)

avg_time = sum(times) / len(times)
max_time = max(times)
min_time = min(times)

print(f"  100 requests:")
print(f"    Average: {avg_time:.1f}ms")
print(f"    Min: {min_time:.1f}ms")
print(f"    Max: {max_time:.1f}ms")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)
print(f"  Total tests: {total}")
print(f"  Correct: {correct}")
print(f"  Accuracy: {accuracy:.1f}%")
print(f"  Average response: {avg_time:.1f}ms")
print(f"  Status: {'PASS ✓' if accuracy >= 90 and avg_time < 100 else 'FAIL ✗'}")
print("\nArchitecture:")
print("  Python ML API: URL lexical features only (20 dimensions)")
print("  Response format:")
print("    {")
print('      "url": "http://example.com",')
print('      "is_phishing": false,')
print('      "probability": 0.15,')
print('      "risk_level": "LOW",')
print('      "features": {...},')
print('      "processing_time_ms": 5')
print("    }")
print("\nJava Backend Responsibility:")
print("  - Receives this response from Python API")
print("  - Decides whether to trigger deep analysis (LLM)")
print("  - Example logic:")
print("    if probability > 0.7 or risk_level == 'CRITICAL':")
print("        # Block immediately, show warning")
print("    elif probability > 0.4:")
print("        # Trigger LLM deep analysis")
print("    else:")
print("        # Allow access")
print("=" * 70)
