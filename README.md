# 🌾 Crop Yield Prediction Dashboard

Dashboard Streamlit interaktif yang mereplikasi alur **CRISP-DM** dari notebook
`crop_yield_crispdm.ipynb`: Business Understanding → Data Understanding →
Data Preparation → Modelling → Evaluation → Deployment (simulasi prediksi).

Dataset sumber: [Crop Yield Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset/data)

## Struktur Folder

```
crop_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── crop_yield_crispdm.ipynb   ← notebook asli (bisa didownload dari dashboard)
├── data/
│   └── yield_df.csv          ← taruh dataset di sini
└── results/
    ├── notebook_metrics.json  ← metrik hasil run notebook asli
    └── figures/                ← grafik hasil mining dari notebook asli
        ├── 01_distribusi_yield.png
        ├── 02_rata_rata_per_item.png
        ├── 03_top15_wilayah.png
        ├── 04_korelasi_numerik.png
        ├── 05_before_after_log.png
        └── 06_actual_vs_predicted_rf.png
```

## Cara Menjalankan

1. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```

2. Download dataset dari Kaggle, lalu letakkan `yield_df.csv` di dalam
   folder `data/` (jika folder ini kosong, dashboard tetap bisa dipakai
   lewat opsi upload manual di sidebar).

3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

4. Browser akan terbuka otomatis (biasanya di `http://localhost:8501`).
   Dashboard akan otomatis membaca `data/yield_df.csv` tanpa perlu upload lagi.

## Struktur Halaman

| Halaman | Isi |
|---|---|
| 🏠 Business Understanding | Latar belakang, tujuan, dan batasan proyek |
| 🔎 Data Understanding | Ringkasan dataset, distribusi yield, top crop/wilayah, korelasi |
| 🛠️ Data Preparation | Transformasi log, split train/test berbasis tahun, baseline, feature engineering |
| 🤖 Modelling & Evaluation | Perbandingan Linear Regression / Random Forest / XGBoost (MAE, RMSE, R²), actual vs predicted |
| 🧩 Feature Importance | Fitur paling berpengaruh dari model berbasis tree |
| 📁 Hasil Mining (Notebook) | Arsip gambar & metrik hasil run asli dari `crop_yield_crispdm.ipynb` |
| 🎯 Simulasi Prediksi | Form input interaktif untuk memprediksi hasil panen pada kondisi baru |

## Catatan

- Model dilatih **saat aplikasi pertama kali dijalankan** (hasil di-cache oleh
  Streamlit dengan `@st.cache_resource`, jadi hanya lambat di run pertama).
- Pendekatan modelling: model memprediksi **residual** terhadap baseline
  rata-rata `log_yield` per kombinasi Area+Item (sama seperti di notebook),
  bukan memprediksi yield secara langsung.
- Jika `xgboost` tidak berhasil ter-install di environment kamu, dashboard
  tetap berjalan normal dengan Linear Regression & Random Forest saja.
