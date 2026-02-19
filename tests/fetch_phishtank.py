# -*- coding: utf-8 -*-
"""
Fetch Real Phishing URLs from PhishTank
从 PhishTank 获取真实钓鱼 URL
"""
import requests
import time
import csv
import os
from datetime import datetime

print("=" * 70)
print("Fetching Real Phishing URLs from PhishTank")
print("=" * 70)

# ============================================================================
# PhishTank API 配置
# ============================================================================
PHISHTANK_API_URL = "https://checkurl.phishtank.com/api/v1"
PHISHTANK_DOWNLOAD_URL = "https://data.phishtank.com/data/"

# 注意：使用 PhishTank API 需要注册获取 API 密钥
# 这里先尝试使用公开的数据集
PHISHTANK_CSV_URL = "https://data.phishtank.com/data/online-valid.csv"

DATA_DIR = "D:/BrowserDownload/FishingSource"
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# 方案 1: 下载 PhishTank 公开 CSV 数据集
# ============================================================================
print("\n[Method 1] Downloading PhishTank public CSV dataset...")

try:
    print(f"  Downloading from: {PHISHTANK_CSV_URL}")
    response = requests.get(PHISHTANK_CSV_URL, timeout=30)

    if response.status_code == 200:
        # 保存原始CSV
        csv_path = os.path.join(DATA_DIR, "phishtank_urls.csv")
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(response.text)

        # 解析URL
        urls = []
        for line in response.text.split('\n')[1:]:  # 跳过表头
            line = line.strip()
            if line and ',' in line:
                # CSV格式: url, ...
                url = line.split(',')[0].strip('"')
                if url and url.startswith('http'):
                    urls.append(url)

        print(f"  Downloaded {len(urls)} phishing URLs")
        print(f"  Saved to: {csv_path}")

        # 保存为简单列表
        list_path = os.path.join(DATA_DIR, "phishtank_urls.txt")
        with open(list_path, 'w', encoding='utf-8') as f:
            for url in urls[:10000]:  # 限制10000个
                f.write(url + '\n')

        print(f"  Saved {min(10000, len(urls))} URLs to: {list_path}")

    else:
        print(f"  Failed: HTTP {response.status_code}")

except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# 方案 2: 使用 OpenPhish 数据集（另一个公明数据源）
# ============================================================================
print("\n[Method 2] Trying OpenPhish dataset...")

OPENPHISH_URL = "https://raw.githubusercontent.com/OpenPhish/Phish.AI/master/verified_credentials_online.csv"

try:
    print(f"  Downloading from: {OPENPHISH_URL}")
    response = requests.get(OPENPHISH_URL, timeout=30)

    if response.status_code == 200:
        urls = []
        for line in response.text.split('\n')[1:]:  # 跳过表头
            line = line.strip()
            if line and ',' in line:
                url = line.split(',')[0].strip('"')
                if url and url.startswith('http'):
                    urls.append(url)

        print(f"  Downloaded {len(urls)} phishing URLs")

        list_path = os.path.join(DATA_DIR, "openphish_urls.txt")
        with open(list_path, 'w', encoding='utf-8') as f:
            for url in urls[:10000]:
                f.write(url + '\n')

        print(f"  Saved {min(10000, len(urls))} URLs to: {list_path}")

    else:
        print(f"  Failed: HTTP {response.status_code}")

except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# 方案 3: 生成合法 URL 列表（来自 Alexa Top 1M 或类似）
# ============================================================================
print("\n[Method 3] Generating legitimate URLs...")

LEGITIMATE_DOMAINS = [
    # Top tech sites
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'microsoft.com',
    'apple.com', 'netflix.com', 'spotify.com', 'twitter.com', 'linkedin.com',
    'instagram.com', 'whatsapp.com', 'telegram.org', 'discord.com', 'reddit.com',
    'github.com', 'stackoverflow.com', 'wikipedia.org', 'yahoo.com', 'ebay.com',

    # Top news sites
    'cnn.com', 'bbc.com', 'nytimes.com', 'theguardian.com', 'washingtonpost.com',
    'foxnews.com', 'nbcnews.com', 'abcnews.go.com', 'reuters.com', 'apnews.com',

    # Top shopping sites
    'walmart.com', 'target.com', 'bestbuy.com', 'homedepot.com', 'lowes.com',
    'etsy.com', 'shopify.com', 'aliexpress.com', 'shein.com', 'newegg.com',

    # Top financial sites
    'chase.com', 'bankofamerica.com', 'wellsfargo.com', 'citibank.com', 'capitalone.com',

    # Top entertainment
    'twitch.tv', 'pinterest.com', 'snap.com', 'tiktok.com', 'vmware.com',
]

legitimate_urls = []

for domain in LEGITIMATE_DOMAINS:
    # 为每个域名生成多个URL
    base_urls = [
        f'https://www.{domain}/',
        f'https://www.{domain}/search',
        f'https://{domain}/',
        f'https://{domain}/about',
        f'https://{domain}/help',
        f'https://www.{domain}/user/profile',
        f'https://www.{domain}/products',
        f'https://{domain}/docs',
    ]
    legitimate_urls.extend(base_urls)

# 去重
import random
random.shuffle(legitimate_urls)

list_path = os.path.join(DATA_DIR, "legitimate_urls.txt")
with open(list_path, 'w', encoding='utf-8') as f:
    for url in legitimate_urls:
        f.write(url + '\n')

print(f"  Generated {len(legitimate_urls)} legitimate URLs")
print(f"  Saved to: {list_path}")

# ============================================================================
# 检查数据文件
# ============================================================================
print("\n[Data Files Summary]")

for filename in ['phishtank_urls.txt', 'openphish_urls.txt', 'legitimate_urls.txt']:
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"  {filename}: {count} URLs")

print("\n" + "=" * 70)
print("Data collection complete!")
print("=" * 70)
