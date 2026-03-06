# 🪙 MLOps-BTCPricePredict

> **Sistem Prediksi Arah Pergerakan Harga Bitcoin (BTC) Berbasis Data Harian**  
> Mata Kuliah Machine Learning Operations — Universitas Brawijaya 2025  
> Dosen: Rizal Setya Perdana, S.Kom., M.Kom., Ph.D.  
> Mahasiswa: Joshua Dwiputra Rendro Joelaskoro (235150201111071)

---

## 📌 Tujuan Proyek

Proyek ini membangun sistem prediksi **arah pergerakan harga harian Bitcoin (naik / turun)** menggunakan pendekatan MLOps dengan fokus pada **Continuous Training (CT)**. Model tidak diperlakukan sebagai artefak statis, melainkan sebagai komponen dinamis yang diperbarui secara berkala mengikuti perubahan distribusi pasar (*concept drift* dan *data drift*).

**Dua tujuan utama:**
1. Membangun sistem prediksi arah harga BTC harian berbasis klasifikasi biner.
2. Merancang mekanisme *continuous training* agar performa model tetap relevan terhadap kondisi pasar terkini.

---

## 🗂️ Struktur Direktori
````
MLOps-BTCPricePredict/
│
├── .devcontainer/
│   └── devcontainer.json        # Konfigurasi GitHub Codespaces
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml               # GitHub Actions CI (lint + test)
│   └── pull_request_template.md
│
├── config/
│   └── config.yaml              # Konfigurasi proyek
│
├── data/
│   ├── raw/                     # Data mentah dari CoinGecko API
│   ├── processed/               # Data hasil feature engineering
│   └── external/                # Data pendukung eksternal
│
├── models/
│   ├── trained/                 # Model hasil training
│   └── registry/                # Model registry
│
├── notebooks/                   # Jupyter Notebooks untuk EDA
│
├── src/
│   ├── data/
│   │   └── ingestion.py         # Fetch data dari CoinGecko
│   ├── features/
│   │   └── engineering.py       # Feature engineering
│   ├── models/
│   │   └── train.py             # Training pipeline
│   ├── monitoring/
│   │   └── monitor.py           # Deteksi drift & trigger retraining
│   └── __init__.py
│
├── tests/                       # Unit tests
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
````

Struktur mengikuti konvensi **Cookiecutter Data Science** yang umum digunakan di industri MLOps.

---

## 🚀 Cara Menjalankan dengan GitHub Codespaces

### 1. Buka Codespaces
1. Klik tombol **`<> Code`** di halaman repositori GitHub.
2. Pilih tab **`Codespaces`**.
3. Klik **`Create codespace on main`**.

Codespaces akan otomatis:
- Menyiapkan Python 3.11
- Menginstal seluruh dependensi dari `requirements.txt`
- Mengaktifkan ekstensi VS Code (Jupyter, Pylance, GitLens)
- Mengekspos port 8888 (Jupyter) dan 5000 (Model API)

### 2. Jalankan Jupyter Notebook
````bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
````

### 3. Fetch data dari CoinGecko
````bash
python src/data/ingestion.py
````

### 4. Jalankan feature engineering
````bash
python src/features/engineering.py
````

---

## 🌿 Branching Strategy — GitHub Flow

Proyek ini menerapkan **GitHub Flow**:
````
main
 └── feat/initial-eda          ← Eksplorasi & analisis data awal
 └── feat/feature-engineering  ← Pipeline feature engineering
 └── feat/model-training       ← Implementasi training pipeline
 └── feat/monitoring           ← Monitoring & drift detection
````

**Aturan:**
- `main` selalu dalam kondisi stabil dan siap dijalankan.
- Setiap fitur / eksperimen dikerjakan di branch tersendiri.
- Merge ke `main` **hanya** melalui Pull Request setelah review.
- Branch eksperimen pertama: `feat/initial-eda`

---

## 📊 ML Task & Arsitektur Pipeline

| Komponen | Detail |
|---|---|
| **Task** | Binary Classification (naik=1 / turun=0) |
| **Data Source** | CoinGecko Public API (harga harian BTC-USD) |
| **Fitur Utama** | Return harian, Moving Average, Volatility rolling, Volume ratio |
| **Algoritma** | Random Forest / Logistic Regression |
| **Training Strategy** | Rolling Window (2 tahun terakhir) |
| **Metrik Utama** | F1-score (threshold ≥ 0.60) |
| **CT Trigger** | F1 drop > 5%, volatility spike, feature drift (KS test) |

---

## 📦 Dependensi Utama

| Library | Kegunaan |
|---|---|
| `pandas`, `numpy` | Manipulasi data |
| `scikit-learn` | Modeling & evaluasi |
| `requests` | Fetch API |
| `evidently` | Drift detection |
| `mlflow` | Experiment tracking |
| `scipy` | KS test untuk distribusi |

Install semua dependensi:
````bash
pip install -r requirements.txt
````

---

## 📝 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE).