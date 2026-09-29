# 🌾 Smart Harvest Planner

Aplikasi web interaktif berbasis **Streamlit** untuk memprediksi hasil panen (crop yield) berdasarkan kondisi agroklimat yang dimasukkan pengguna. Model prediksi dibangun menggunakan algoritma **Random Forest** dan dikembangkan mengikuti alur metodologi **CRISP-DM** di notebook `crop_yield_crispdm.ipynb`.

Dataset sumber: [Crop Yield Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset/data)

---

## Struktur Folder

```
crop_yield_dashboard/
├── app.py                        ← Logika utama aplikasi Streamlit
├── utils.py                      ← Helper: load CSS, render HTML template
├── requirements.txt              ← Daftar dependency Python
├── README.md
├── crop_yield_crispdm.ipynb      ← Notebook CRISP-DM (eksplorasi & training model)
├── assets/
│   ├── style.css                 ← Seluruh styling CSS halaman
│   ├── templates.html            ← Komponen HTML (hero banner, kartu metrik, DSS)
│   ├── logo.png                  ← Logo aplikasi (ukuran besar)
│   └── logo_small.png            ← Logo aplikasi (favicon / ukuran kecil)
├── data/
│   └── yield_df.csv              ← Dataset historis hasil panen (dari Kaggle)
└── models/
    └── rf_model.pkl              ← Model Random Forest yang sudah dilatih (via notebook)
```

---

## Cara Menjalankan

### 1. Clone atau unduh repository ini

```bash
git clone <url-repository>
cd crop_yield_dashboard
```

### 2. Install dependency

Disarankan menggunakan virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 3. Pastikan file berikut tersedia

| File | Lokasi | Keterangan |
|---|---|---|
| `yield_df.csv` | `data/yield_df.csv` | Dataset historis dari Kaggle |
| `rf_model.pkl` | `models/rf_model.pkl` | Model hasil training di notebook |

> **Penting:** File `rf_model.pkl` harus dihasilkan terlebih dahulu dengan menjalankan seluruh cell di `crop_yield_crispdm.ipynb` hingga bagian **Deployment**. Model akan tersimpan otomatis ke folder `models/`.

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Browser akan terbuka otomatis di `http://localhost:8501`.

---

## Cara Menggunakan Aplikasi

Aplikasi terdiri dari satu halaman utama dengan tiga bagian output:

### 📝 Form Input Parameter
Isi enam parameter berikut, lalu klik **"Jalankan Prediksi"**:

| No. | Parameter | Keterangan |
|---|---|---|
| 1 | **Wilayah (Negara)** | Pilih negara dari daftar yang tersedia |
| 2 | **Komoditas Utama** | Pilih jenis tanaman (10 komoditas tersedia) |
| 3 | **Tahun Proyeksi** | Pilih tahun masa depan (2027 – 2045) |
| 4 | **Curah Hujan (mm/tahun)** | Rata-rata curah hujan tahunan |
| 5 | **Suhu Rata-rata (°C)** | Rata-rata suhu tahunan |
| 6 | **Penggunaan Pestisida (Ton)** | Total pestisida yang digunakan |

### 📊 Output 1 — Prediksi Utama
Menampilkan estimasi hasil panen untuk komoditas yang dipilih dalam satuan **ton/ha** dan **hg/ha** (standar FAO), disertai informasi lokasi dan tahun proyeksi.

### 🔍 Output 2 — Analisis Sensitivitas
Menampilkan simulasi tiga skenario perubahan kondisi iklim secara otomatis:
- Suhu naik +1°C
- Curah hujan turun -10%
- Pestisida naik +20%

Setiap skenario menampilkan estimasi yield baru dan persentase perubahan dibandingkan prediksi utama.

### 🏆 Output 3 — Perangkingan & Decision Support System (DSS)
Menjalankan prediksi untuk **semua 10 komoditas** secara bersamaan lalu merangkingnya, dan memberikan rekomendasi komoditas paling produktif untuk kondisi yang dimasukkan, dilengkapi grafik batang horizontal.

---

## Komoditas yang Didukung

| Nama Indonesia | Nama Inggris (kode model) |
|---|---|
| Jagung | Maize |
| Kentang | Potatoes |
| Padi | Rice, paddy |
| Sorgum | Sorghum |
| Kedelai | Soybeans |
| Gandum | Wheat |
| Singkong / Ubi Kayu | Cassava |
| Ubi | Yams |
| Ubi Jalar | Sweet potatoes |
| Pisang & Lainnya | Plantains and others |

---

## Catatan Teknis

- **Model tidak dilatih ulang saat aplikasi dijalankan.** File `rf_model.pkl` sudah berisi model yang terlatih. Aplikasi hanya memuat model tersebut menggunakan `joblib.load` — proses ini di-cache oleh Streamlit (`@st.cache_resource`) sehingga hanya berjalan sekali saat pertama kali dibuka.
- **Pendekatan prediksi:** Model memprediksi *residual* terhadap baseline rata-rata `log_yield` per kombinasi Area + Item. Hasil akhir dikembalikan ke satuan asli menggunakan `expm1`.
- **Pemisahan kode:** CSS ada di `assets/style.css`, komponen HTML di `assets/templates.html`, dan `app.py` hanya berisi logika Python. Helper untuk membaca keduanya ada di `utils.py`.
- **Data referensi** (`baseline_map`, `area_list`, `item_columns`) dihitung langsung dari `yield_df.csv` saat runtime — tidak ada file `.pkl` tambahan selain model.
