"""
Download datasets from original GitHub sources
"""
import os
import urllib.request
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"  [SKIP] {filename} already exists")
        return True
    try:
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath) / 1024
        print(f"  [OK] {filename} ({size:.1f} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {filename}: {e}")
        return False

# ==========================================
# 1. PhoNER_COVID19 - from VinAI GitHub
# ==========================================
print("="*60)
print("1. PhoNER_COVID19 (Vietnamese NER)")
print("   Source: github.com/VinAIResearch/PhoNER_COVID19")
print("="*60)

PHONER_BASE = "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data"
for split in ["train", "dev", "test"]:
    download_file(
        f"{PHONER_BASE}/word/{split}_word.conll",
        f"PhoNER_COVID19_{split}.conll"
    )

# ==========================================
# 2. UIT-ViSFD from available sources
# ==========================================
print("\n" + "="*60)
print("2. UIT-ViSFD (Smartphone Reviews - Aspect Sentiment)")
print("   Trying multiple sources...")
print("="*60)

# Try from nttuan8 (common mirror)
VISFD_SOURCES = [
    "https://raw.githubusercontent.com/kimkim00/UIT-ViSFD/main/Train.csv",
    "https://raw.githubusercontent.com/kimkim00/UIT-ViSFD/main/Dev.csv", 
    "https://raw.githubusercontent.com/kimkim00/UIT-ViSFD/main/Test.csv",
]
visfd_ok = True
for url in VISFD_SOURCES:
    fname = "UIT-ViSFD_" + url.split("/")[-1]
    if not download_file(url, fname):
        visfd_ok = False

if not visfd_ok:
    # Try alternative
    print("  Trying HuggingFace direct files...")
    HF_BASE = "https://huggingface.co/datasets/SEACrowd/uit_visfd/resolve/main"
    for fname in ["train.csv", "dev.csv", "test.csv"]:
        download_file(f"{HF_BASE}/{fname}", f"UIT-ViSFD_{fname}")

# ==========================================
# 3. UIT-VSFC - Vietnamese Students Feedback
# ==========================================
print("\n" + "="*60)
print("3. UIT-VSFC (Vietnamese Feedback Sentiment)")
print("="*60)

VSFC_SOURCES = [
    ("https://raw.githubusercontent.com/Insight-AI-Lab/UIT-VSFC/main/data/train_sents.txt", "UIT-VSFC_train_sents.txt"),
    ("https://raw.githubusercontent.com/Insight-AI-Lab/UIT-VSFC/main/data/train_sentiments.txt", "UIT-VSFC_train_sentiments.txt"),
    ("https://raw.githubusercontent.com/Insight-AI-Lab/UIT-VSFC/main/data/dev_sents.txt", "UIT-VSFC_dev_sents.txt"),
    ("https://raw.githubusercontent.com/Insight-AI-Lab/UIT-VSFC/main/data/dev_sentiments.txt", "UIT-VSFC_dev_sentiments.txt"),
    ("https://raw.githubusercontent.com/Insight-AI-Lab/UIT-VSFC/main/data/test_sents.txt", "UIT-VSFC_test_sents.txt"),
    ("https://raw.githubusercontent.com/Insight-AI-Lab/UIT-VSFC/main/data/test_sentiments.txt", "UIT-VSFC_test_sentiments.txt"),
]
for url, fname in VSFC_SOURCES:
    download_file(url, fname)

# ==========================================
# 4. Vietnamese Stopwords
# ==========================================
print("\n" + "="*60)
print("4. Vietnamese Stopwords")
print("="*60)
download_file(
    "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt",
    "vietnamese_stopwords.txt"
)

# ==========================================
# Summary
# ==========================================
print("\n" + "="*60)
print("DOWNLOAD SUMMARY")
print("="*60)

files = sorted([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
total = sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in files)
print(f"Total: {len(files)} files, {total/1024/1024:.2f} MB\n")
for f in files:
    sz = os.path.getsize(os.path.join(DATA_DIR, f))
    print(f"  {f}: {sz/1024:.1f} KB")
