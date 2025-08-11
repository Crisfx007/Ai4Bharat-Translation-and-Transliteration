# AI4Bharat Translation & Transliteration Pipeline

This repository provides a pipeline for **detecting, transliterating, and translating Indian languages** (including Hinglish) into English using **AI4Bharat’s transliteration tools** and **Meta’s NLLB-200 translation model**.

The workflow:
- Detects language/script (with script-based fallback).
- Handles Romanized Hindi (Hinglish) by converting to Devanagari before translation.
- Translates supported Indian languages into English.
- Splits large datasets into smaller parts for processing.

---

## 🚀 Features
- **Multi-language detection** using CLD3 and script fallback.
- **Hinglish → Hindi** transliteration before translation.
- **GPU acceleration** if CUDA is available.
- **Automatic dataset chunking** for large inputs.
- **Configurable** number of parts and ranges.

---

## 📋 Requirements
- **Linux environment** (or WSL on Windows).
- Python **3.8+**.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) *(recommended for dependency management)*.
- All dependencies listed in `transliterate.yml`.

---

## ⚙️ Installation

### 1️⃣ Install Miniconda (if not installed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
