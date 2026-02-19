# -*- coding: utf-8 -*-
"""
Generate Realistic Phishing URL Dataset
"""
import random
import csv
import os

print("=" * 70)
print("NoFishing - Realistic Dataset Generation")
print("=" * 70)

DATA_DIR = "D:/BrowserDownload/FishingSource"
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# Configuration
# ============================================================================
BRANDS = [
    'paypal', 'apple', 'google', 'amazon', 'microsoft',
    'facebook', 'netflix', 'spotify', 'chase', 'wellsfargo',
    'bankofamerica', 'citibank', 'capitalone', 'adobe',
    'dropbox', 'github', 'steam', 'instagram', 'linkedin',
]

FREE_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.cc']

LEGIT_DOMAINS = [
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com',
    'apple.com', 'microsoft.com', 'netflix.com', 'spotify.com',
    'twitter.com', 'linkedin.com', 'github.com', 'stackoverflow.com',
    'reddit.com', 'wikipedia.org', 'yahoo.com', 'ebay.com',
    'paypal.com', 'chase.com', 'wellsfargo.com', 'cnn.com',
]

PHISHING_TEMPLATES = [
    'http://{brand}-verify{tld}/login',
    'http://{brand}-secure{tld}/signin',
    'http://{brand}-account{tld}/verify',
    'http://verify-{brand}{tld}/auth',
    'http://secure-{brand}{tld}/login',
    'http://{ip}/login',
    'http://{ip}/signin',
    'https://{ip}/secure/login',
    'http://{brand}-support.tk/account',
    'http://{brand}-{random}.ml/signin',
]

LEGIT_PATHS = ['', 'search', 'about', 'help', 'products', 'docs',
               'user/profile', 'settings', 'privacy', 'terms']

# ============================================================================
# Generate URLs
# ============================================================================
print("\n[Generating] Creating dataset...")

NUM_PHISHING = 5000
NUM_LEGIT = 5000

data = []

# Generate phishing
print("  Generating phishing URLs...")
for i in range(NUM_PHISHING):
    template = random.choice(PHISHING_TEMPLATES)
    brand = random.choice(BRANDS)
    tld = random.choice(FREE_TLDS)
    ip = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
    rand_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))

    url = template.format(brand=brand, tld=tld, ip=ip, random=rand_str)
    data.append((url, 1))

    if (i + 1) % 1000 == 0:
        print(f"    {i+1}/{NUM_PHISHING}")

# Generate legitimate
print("  Generating legitimate URLs...")
for i in range(NUM_LEGIT):
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)

    if path:
        url = f"https://www.{domain}/{path}"
    else:
        url = f"https://www.{domain}/"

    data.append((url, 0))

    if (i + 1) % 1000 == 0:
        print(f"    {i+1}/{NUM_LEGIT}")

# Shuffle
random.shuffle(data)

# ============================================================================
# Save dataset
# ============================================================================
print("\n[Saving] Writing to files...")

csv_file = os.path.join(DATA_DIR, "realistic_dataset.csv")
with open(csv_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['url', 'label'])
    for url, label in data:
        writer.writerow([url, label])

print(f"  CSV: {csv_file} ({len(data)} samples)")

# Save by class
phishing = [url for url, label in data if label == 1]
legit = [url for url, label in data if label == 0]

with open(os.path.join(DATA_DIR, "phishing_urls.txt"), 'w', encoding='utf-8') as f:
    for url in phishing:
        f.write(url + '\n')

with open(os.path.join(DATA_DIR, "legit_urls.txt"), 'w', encoding='utf-8') as f:
    for url in legit:
        f.write(url + '\n')

print(f"  Phishing: {len(phishing)}")
print(f"  Legitimate: {len(legit)}")

# ============================================================================
# Statistics
# ============================================================================
print("\n[Statistics]")

phish_lens = [len(url) for url, label in data if label == 1]
legit_lens = [len(url) for url, label in data if label == 0]

print(f"  Phishing avg length: {sum(phish_lens)/len(phish_lens):.1f}")
print(f"  Legitimate avg length: {sum(legit_lens)/len(legit_lens):.1f}")

print("\n  Sample phishing:")
for url, label in data[:10]:
    if label == 1:
        print(f"    {url}")

print("\n  Sample legitimate:")
for url, label in data[:10]:
    if label == 0:
        print(f"    {url}")

print("\n" + "=" * 70)
print("Dataset generation complete!")
print("=" * 70)
