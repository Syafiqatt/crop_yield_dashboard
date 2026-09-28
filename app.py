"""
Dashboard Simulasi Prediksi Hasil Panen (Crop Yield Prediction)
Model: Random Forest (rf_model.pkl) — pipeline CRISP-DM
Author: Data Science Project - Syafiq

Catatan struktur:
- CSS ada di assets/style.css
- HTML fragment ada di assets/templates.html
- app.py ini HANYA berisi logika (load data/model, prediksi, orkestrasi UI)
- Hanya 1 file model (rf_model.pkl) yang dipakai; baseline & daftar area/item
  dihitung langsung dari data/yield_df.csv saat runtime, jadi tidak perlu
  simpan baseline_map.pkl / global_mean.pkl / item_columns.pkl / dll.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import textwrap
import base64
from utils import get_image_base64, load_css, render_template

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"
DATA_PATH  = BASE_DIR / "data" / "yield_df.csv"
CSS_PATH   = BASE_DIR / "assets" / "style.css"
TPL_PATH   = BASE_DIR / "assets" / "templates.html"
LOGO_PATH  = BASE_DIR / "assets" / "logo_small.png"
BANNER_PATH = BASE_DIR / "assets" / "header_banner.png"

CROP_TRANSLATION = {
    "Maize": "Jagung (Maize)",
    "Potatoes": "Kentang (Potatoes)",
    "Rice, paddy": "Padi (Rice, paddy)",
    "Sorghum": "Sorgum (Sorghum)",
    "Soybeans": "Kedelai (Soybeans)",
    "Wheat": "Gandum (Wheat)",
    "Cassava": "Singkong / Ubi Kayu (Cassava)",
    "Yams": "Ubi (Yams)",
    "Sweet potatoes": "Ubi Jalar (Sweet potatoes)",
    "Plantains and others": "Pisang & Lainnya (Plantains)",
}
CROP_REVERSE = {v: k for k, v in CROP_TRANSLATION.items()}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS EKSTERNAL
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSmart — Simulasi Prediksi Hasil Panen",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_css(CSS_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & DATA REFERENSI (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model Random Forest...")
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"⚠️ Model tidak ditemukan: `{MODEL_PATH.name}`. Jalankan cell Deployment di notebook terlebih dahulu.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Memuat data historis...")
def load_reference_data():
    if not DATA_PATH.exists():
        st.error(f"⚠️ Dataset tidak ditemukan: `{DATA_PATH.name}`.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["log_yield"] = np.log1p(df["hg/ha_yield"])
    global_mean_ = df["log_yield"].mean()
    baseline_map_ = df.groupby(["Area", "Item"])["log_yield"].mean().to_dict()
    area_list_ = sorted(df["Area"].unique().tolist())
    raw_item_list_ = sorted(df["Item"].unique().tolist())
    item_columns_ = list(pd.get_dummies(df["Item"], prefix="item").columns)
    return baseline_map_, global_mean_, area_list_, raw_item_list_, item_columns_


model = load_model()
baseline_map, global_mean, area_list, raw_item_list, item_columns = load_reference_data()

# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
def predict_single(area: str, item_raw: str, year: int, rain: float, temp: float, pesticide: float):
    rainfall_temp = rain * temp
    row = {
        "Year": year,
        "average_rain_fall_mm_per_year": rain,
        "pesticides_tonnes": pesticide,
        "avg_temp": temp,
        "rainfall_temp": rainfall_temp,
    }
    item_data = {col: 0 for col in item_columns}
    target_col = f"item_{item_raw}"
    if target_col in item_data:
        item_data[target_col] = 1

    x_df = pd.DataFrame([{**row, **item_data}])
    pred_res = model.predict(x_df)[0]
    base_log = baseline_map.get((area, item_raw), global_mean)
    pred_log = base_log + pred_res
    pred_hg = float(np.expm1(pred_log))
    pred_ton = max(0.0, pred_hg / 10_000)
    return pred_ton, pred_hg


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER (KUSTOM HTML & CSS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    render_template(TPL_PATH, "hero"),
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# FORM INPUT — BERBENTUK KEBAWAH (SESUAI GAMBAR 2)
# ─────────────────────────────────────────────────────────────────────────────
items_indo = [CROP_TRANSLATION.get(c, c) for c in raw_item_list]
default_area_idx = area_list.index("Indonesia") if "Indonesia" in area_list else 0
default_item_idx = next((i for i, x in enumerate(items_indo) if "Padi" in x), 0)

with st.container():
    with st.form("input_form_main"):
        st.markdown("<h3 style='color: #2E7D32; font-weight: 700; margin-top: 0; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 8px;'>📝 Parameter Agroklimat</h3>", unsafe_allow_html=True)
        
        area = st.selectbox("1. Wilayah (Negara)", area_list, index=default_area_idx)
        item_indo = st.selectbox("2. Komoditas Utama", items_indo, index=default_item_idx)
        year = st.slider("3. Tahun Pengamatan/Proyeksi:", 1990, 2030, 2013)
        rain = st.number_input("4. Curah Hujan (mm/tahun)", value=1485.0, step=10.0)
        temp = st.number_input("5. Suhu Rata-rata (°C)", value=16.37, step=0.1, format="%.2f")
        pesticide = st.number_input("6. Penggunaan Pestisida (Ton)", value=121.0, step=1.0)

        submitted = st.form_submit_button("Jalankan Prediksi", use_container_width=True)

item_raw = CROP_REVERSE.get(item_indo, item_indo)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 1 — PREDIKSI UTAMA
# ─────────────────────────────────────────────────────────────────────────────
y_ton, y_hg = predict_single(area, item_raw, year, rain, temp, pesticide)

st.markdown(
    render_template(TPL_PATH, "section_title", number="1", icon="📊", title="Prediksi Utama Komoditas Pilihan"),
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(render_template(TPL_PATH, "metric_card", box_class="green-box",
                                 title="Estimasi Hasil Panen", value=f"{y_ton:,.2f} ton/ha"), unsafe_allow_html=True)
with c2:
    st.markdown(render_template(TPL_PATH, "metric_card", box_class="blue-box",
                                 title="Satuan Standar FAO", value=f"{y_hg:,.0f} hg/ha"), unsafe_allow_html=True)
with c3:
    st.markdown(render_template(TPL_PATH, "metric_card", box_class="info-box",
                                 title="Lokasi & Tahun", value=f"{area} ({year})"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 2 — ANALISIS SENSITIVITAS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    render_template(TPL_PATH, "section_title", number="2", icon="🔍",
                     title="Analisis Sensitivitas Iklim & Pestisida (Uji Skenario)"),
    unsafe_allow_html=True,
)

y_temp_up, _ = predict_single(area, item_raw, year, rain, temp + 1.0, pesticide)
d_temp = ((y_temp_up - y_ton) / y_ton * 100) if y_ton > 0 else 0.0

y_rain_down, _ = predict_single(area, item_raw, year, rain * 0.9, temp, pesticide)
d_rain = ((y_rain_down - y_ton) / y_ton * 100) if y_ton > 0 else 0.0

y_pest_up, _ = predict_single(area, item_raw, year, rain, temp, pesticide * 1.2)
d_pest = ((y_pest_up - y_ton) / y_ton * 100) if y_ton > 0 else 0.0

scenarios = [
    ("🌡️ Suhu Naik (+1°C)", y_temp_up, d_temp),
    ("🌧️ Hujan Turun (-10%)", y_rain_down, d_rain),
    ("🧪 Pestisida Naik (+20%)", y_pest_up, d_pest),
]
s1, s2, s3 = st.columns(3)
for col, (title, val, diff) in zip((s1, s2, s3), scenarios):
    with col:
        st.markdown(render_template(
            TPL_PATH, "sensitivity_card",
            title=title, value=f"{val:,.2f}",
            badge_class="badge-pos" if diff >= 0 else "badge-neg",
            sign="+" if diff >= 0 else "",
            diff=f"{diff:.2f}",
        ), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 3 — PERANGKINGAN KOMODITAS & DSS
# ─────────────────────────────────────────────────────────────────────────────
rank_df = pd.DataFrame(rankings).sort_values("yield_ton", ascending=False)

fig = go.Figure(go.Bar(
    x=rank_df["crop"],
    y=rank_df["yield_ton"],
    marker=dict(
        color="rgba(129, 199, 132, 0.85)",
        line=dict(color="#2E7D32", width=1.5)
    ),
    text=rank_df["yield_ton"].round(2),
    textposition="outside",
))

fig.update_layout(
    height=450,
    margin=dict(l=10, r=20, t=30, b=80),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    xaxis=dict(
        title="Komoditas",
        tickfont=dict(size=12),
    ),
    yaxis=dict(
        title="Hasil Panen (ton/ha)",
        gridcolor="#e2e8f0",
    ),
    font=dict(family="Outfit"),
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displayModeBar": False}
)