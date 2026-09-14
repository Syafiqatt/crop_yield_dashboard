"""
Dashboard Simulasi Prediksi Hasil Panen (Crop Yield Prediction)
Memuat model & artefak dari crop_yield_crispdm.ipynb
Author: Data Science Project - Syafiq
Theme: White & Green Agricultural Aesthetics
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Prediction — Simulasi Prediksi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Fresh White & Green Agriculture Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        color: #1e293b;
    }

    .stApp {
        background: linear-gradient(180deg, #f0fdf4 0%, #f8faf9 25%, #f4fbf6 70%, #edf7f0 100%);
        background-attachment: fixed;
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #14532d 0%, #166534 35%, #15803d 70%, #16a34a 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 22px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 16px 40px rgba(22, 101, 52, 0.22), 0 4px 12px rgba(22, 101, 52, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 60%);
        animation: subtle-pulse 6s ease-in-out infinite;
    }
    @keyframes subtle-pulse {
        0%, 100% { transform: scale(1); opacity: 0.6; }
        50% { transform: scale(1.08); opacity: 0.9; }
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #f0fdf4 !important;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        margin-bottom: 0.75rem;
        position: relative;
        z-index: 1;
    }
    .hero-banner h1 {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    .hero-banner p {
        font-size: 1.02rem;
        color: rgba(240, 253, 244, 0.9) !important;
        position: relative;
        z-index: 1;
        margin: 0;
        line-height: 1.6;
    }

    /* ── Cards & Containers ── */
    .agri-card {
        background: #ffffff;
        border: 1px solid #dcfce7;
        border-radius: 18px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(22, 101, 52, 0.05), 0 1px 3px rgba(0,0,0,0.02);
        transition: all 0.3s ease;
    }
    .agri-card:hover {
        border-color: #86efac;
        box-shadow: 0 8px 30px rgba(22, 101, 52, 0.1);
    }

    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #166534 !important;
        letter-spacing: -0.2px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .form-group-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #15803d;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* ── Form Inputs ── */
    .stSelectbox label, .stNumberInput label {
        color: #334155 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #f8faf9 !important;
        border-color: #d1fae5 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #10b981 !important;
    }
    div[data-baseweb="input"] {
        background-color: #f8faf9 !important;
        border-color: #d1fae5 !important;
        border-radius: 10px !important;
    }

    /* ── Submit Button ── */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #15803d 0%, #16a34a 50%, #22c55e 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        width: 100% !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 8px 24px rgba(22, 163, 74, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #166534 0%, #15803d 50%, #16a34a 100%) !important;
        box-shadow: 0 12px 30px rgba(22, 163, 74, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Result Cards ── */
    .result-card {
        background: linear-gradient(145deg, #ffffff 0%, #f0fdf4 100%);
        border: 1.5px solid #bbf7d0;
        border-radius: 18px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(22, 101, 52, 0.07);
        transition: all 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(22, 101, 52, 0.14);
        border-color: #4ade80;
    }
    .result-card .label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #166534 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .result-card .value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #14532d !important;
        letter-spacing: -0.5px;
        line-height: 1.15;
    }
    .result-card .unit {
        font-size: 0.82rem;
        font-weight: 500;
        color: #64748b !important;
        margin-top: 0.35rem;
    }

    .confidence-bar-outer {
        background: #e2e8f0;
        border-radius: 10px;
        height: 8px;
        margin-top: 0.6rem;
        overflow: hidden;
    }
    .confidence-bar-inner {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }

    /* ── Chart & Sub-Containers ── */
    .chart-box {
        background: #ffffff;
        border: 1px solid #e2ece5;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 2px 12px rgba(22, 101, 52, 0.04);
        margin-bottom: 1rem;
    }

    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(22, 163, 74, 0.25), transparent);
        margin: 2rem 0;
    }

    .warn-badge {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #92400e !important;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .info-badge {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #166534 !important;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    #MainMenu, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

MODEL_PATH       = BASE_DIR / "rf_model.pkl"
BASELINE_PATH    = BASE_DIR / "baseline_map.pkl"
GLOBAL_MEAN_PATH = BASE_DIR / "global_mean.pkl"
ITEM_COLS_PATH   = BASE_DIR / "item_columns.pkl"
AREA_LIST_PATH   = BASE_DIR / "area_list.pkl"
ITEM_LIST_PATH   = BASE_DIR / "item_list.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model Random Forest...")
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Memuat artefak modeling...")
def load_artifacts():
    baseline_map = joblib.load(BASELINE_PATH)
    global_mean  = float(joblib.load(GLOBAL_MEAN_PATH))
    item_columns = list(joblib.load(ITEM_COLS_PATH))
    area_list    = list(joblib.load(AREA_LIST_PATH))
    item_list    = list(joblib.load(ITEM_LIST_PATH))
    return baseline_map, global_mean, item_columns, area_list, item_list


missing_files = [
    p for p in [MODEL_PATH, BASELINE_PATH, GLOBAL_MEAN_PATH, ITEM_COLS_PATH, AREA_LIST_PATH, ITEM_LIST_PATH]
    if not p.exists()
]
if missing_files:
    st.error(
        "⚠️ File artefak berikut tidak ditemukan:\n\n"
        + "\n".join(f"- `{p.name}`" for p in missing_files)
    )
    st.stop()

model = load_model()
baseline_map, global_mean, item_columns, area_list, item_list = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_yield(area: str, item: str, year: int,
                  rainfall: float, pesticides: float, avg_temp: float):
    key = (area, item)
    if hasattr(baseline_map, 'get'):
        baseline = baseline_map.get(key, global_mean)
    else:
        baseline = global_mean
    used_global = (baseline == global_mean)

    input_feats = pd.DataFrame({
        "Year":                           [float(year)],
        "average_rain_fall_mm_per_year":  [rainfall],
        "pesticides_tonnes":              [pesticides],
        "avg_temp":                       [avg_temp],
        "rainfall_temp":                  [rainfall * avg_temp],
    })

    item_ohe = pd.DataFrame(0, index=[0], columns=item_columns)
    col_key = f"item_{item}"
    if col_key in item_ohe.columns:
        item_ohe[col_key] = 1

    X = pd.concat([input_feats, item_ohe], axis=1)
    residual_pred = model.predict(X)[0]
    log_pred      = baseline + residual_pred
    yield_hg_ha   = float(np.clip(np.expm1(log_pred), 0, None))

    return yield_hg_ha, float(baseline), used_global


def get_confidence(hg_ha: float, area: str, item: str, used_global: bool):
    if used_global:
        return "Sedang", 55, "#d97706"
    baseline_hg = np.expm1(baseline_map.get((area, item), global_mean))
    ratio = hg_ha / max(baseline_hg, 1)
    if 0.5 <= ratio <= 2.0:
        return "Tinggi", 90, "#16a34a"
    elif 0.3 <= ratio <= 3.0:
        return "Cukup", 72, "#0284c7"
    return "Rendah", 42, "#dc2626"


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero-banner'>"
    "<div class='hero-badge'>🌱 Agrikultur & Smart Farming AI</div>"
    "<h1>🌾 Simulasi Prediksi Hasil Panen</h1>"
    "<p>Masukkan parameter wilayah, jenis tanaman, dan kondisi agroklimat untuk memperoleh<br>"
    "estimasi produktivitas panen berbasis model <b>Random Forest</b> dari pipeline CRISP-DM.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='agri-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>🌿 Parameter Input Agroklimat & Komoditas</div>", unsafe_allow_html=True)

with st.form("prediction_form", clear_on_submit=False):
    col_left, col_mid, col_right = st.columns([1.2, 1.2, 1], gap="large")

    with col_left:
        st.markdown("<div class='form-group-title'>📍 Lokasi & Tanaman</div>", unsafe_allow_html=True)
        area = st.selectbox("Wilayah (Area)", sorted(area_list), index=0,
                            help="Pilih negara/wilayah dari data historis.")
        item = st.selectbox("Jenis Tanaman (Item)", sorted(item_list), index=1,
                            help="Jenis tanaman pangan yang ingin diprediksi.")

    with col_mid:
        st.markdown("<div class='form-group-title'>📅 Waktu & Suhu</div>", unsafe_allow_html=True)
        year = st.number_input("Tahun Prediksi", min_value=1990, max_value=2100, value=2025, step=1,
                               help="Tahun target prediksi. Model dilatih pada data 1990–2009.")
        avg_temp = st.number_input("Suhu Rata-rata (°C)", min_value=-10.0, max_value=50.0,
                                   value=22.0, step=0.1, format="%.1f",
                                   help="Suhu rata-rata tahunan dalam derajat Celsius.")

    with col_right:
        st.markdown("<div class='form-group-title'>🌧️ Iklim & Input Tani</div>", unsafe_allow_html=True)
        rainfall = st.number_input("Curah Hujan (mm/tahun)", min_value=0.0, max_value=10000.0,
                                   value=1200.0, step=50.0, format="%.1f",
                                   help="Total curah hujan rata-rata per tahun.")
        pesticides = st.number_input("Pestisida (ton)", min_value=0.0, max_value=1_000_000.0,
                                     value=5000.0, step=100.0, format="%.1f",
                                     help="Total penggunaan pestisida dalam ton.")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🌱  Jalankan Simulasi Prediksi", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION RESULT
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    with st.spinner("Memproses prediksi..."):
        pred_hg_ha, baseline_log, used_global = predict_yield(
            area, item, int(year), rainfall, pesticides, avg_temp
        )

    pred_ton_ha  = pred_hg_ha / 10_000
    baseline_hg  = np.expm1(baseline_log)
    baseline_ton = baseline_hg / 10_000
    delta_pct    = ((pred_hg_ha - baseline_hg) / max(baseline_hg, 1)) * 100

    conf_label, conf_score, conf_color = get_confidence(pred_hg_ha, area, item, used_global)

    # ── Result Cards ──────────────────────────────────────────────────────────
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 Hasil Simulasi Estimasi Panen</div>", unsafe_allow_html=True)

    rc1, rc2, rc3, rc4 = st.columns(4, gap="medium")

    with rc1:
        st.markdown(
            f"<div class='result-card'><div class='label'>Prediksi Yield</div>"
            f"<div class='value'>{pred_ton_ha:,.2f}</div>"
            f"<div class='unit'>ton / hektare</div></div>",
            unsafe_allow_html=True,
        )
    with rc2:
        st.markdown(
            f"<div class='result-card'><div class='label'>Satuan Standar</div>"
            f"<div class='value'>{pred_hg_ha:,.0f}</div>"
            f"<div class='unit'>hg / ha (hektogram)</div></div>",
            unsafe_allow_html=True,
        )
    with rc3:
        sign  = "▲" if delta_pct >= 0 else "▼"
        dcolor = "#16a34a" if delta_pct >= 0 else "#dc2626"
        st.markdown(
            f"<div class='result-card'><div class='label'>vs. Baseline Historis</div>"
            f"<div class='value' style='color:{dcolor} !important;'>{sign} {abs(delta_pct):.1f}%</div>"
            f"<div class='unit'>baseline: {baseline_ton:,.2f} ton/ha</div></div>",
            unsafe_allow_html=True,
        )
    with rc4:
        st.markdown(
            f"<div class='result-card'><div class='label'>Tingkat Kepercayaan</div>"
            f"<div class='value' style='color:{conf_color} !important;'>{conf_label}</div>"
            f"<div class='confidence-bar-outer'>"
            f"<div class='confidence-bar-inner' style='width:{conf_score}%;background:linear-gradient(90deg,{conf_color},{conf_color}bb);'></div>"
            f"</div><div class='unit' style='margin-top:0.4rem;'>{conf_score}% skor model</div></div>",
            unsafe_allow_html=True,
        )

    # ── Detail + Charts ───────────────────────────────────────────────────────
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    detail_col, chart_col = st.columns([1, 1.6], gap="large")

    with detail_col:
        st.markdown("<div class='agri-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📋 Ringkasan Parameter</div>", unsafe_allow_html=True)

        rows_html = ""
        for lbl, val in [
            ("🌍 Wilayah", area), ("🌱 Jenis Tanaman", item), ("📅 Tahun", str(year)),
            ("🌡️ Suhu", f"{avg_temp:.1f} °C"),
            ("🌧️ Curah Hujan", f"{rainfall:,.1f} mm/thn"),
            ("🧪 Pestisida", f"{pesticides:,.1f} ton"),
        ]:
            rows_html += (
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:0.5rem 0;border-bottom:1px solid #f1f5f9;'>"
                f"<span style='color:#64748b;font-size:0.86rem;font-weight:500;'>{lbl}</span>"
                f"<span style='color:#0f172a;font-size:0.86rem;font-weight:700;'>{val}</span></div>"
            )
        st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🔗 Pipeline Dekomposisi</div>", unsafe_allow_html=True)

        residual_val = np.log1p(pred_hg_ha) - baseline_log
        pipe_html = ""
        for lbl, val in [
            ("Baseline log(yield)", f"{baseline_log:.4f}"),
            ("Residual RF", f"{residual_val:+.4f}"),
            ("Log prediksi", f"{np.log1p(pred_hg_ha):.4f}"),
            ("Yield (hg/ha)", f"{pred_hg_ha:,.2f}"),
        ]:
            pipe_html += (
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:0.4rem 0;border-bottom:1px solid #f1f5f9;'>"
                f"<span style='color:#64748b;font-size:0.82rem;'>{lbl}</span>"
                f"<span style='color:#15803d;font-size:0.82rem;font-weight:700;font-family:monospace;'>{val}</span></div>"
            )
        st.markdown(pipe_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if used_global:
            st.markdown(
                f"<div class='warn-badge'>⚠️ Kombinasi <b>{area} + {item}</b> tidak ada di data "
                "historis. Baseline menggunakan rata-rata global.</div>",
                unsafe_allow_html=True,
            )

    with chart_col:
        # Gauge Chart in White/Green Palette
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_ton_ha,
            delta={"reference": baseline_ton, "relative": False, "valueformat": ".2f",
                   "suffix": " ton/ha", "font": {"size": 13, "color": "#64748b"}},
            number={"suffix": " ton/ha", "font": {"size": 32, "color": "#14532d", "family": "Plus Jakarta Sans"},
                    "valueformat": ",.2f"},
            title={"text": f"<b>Estimasi Produktivitas Hasil Panen</b><br>"
                           f"<span style='font-size:12px;color:#64748b'>{item} — {area} ({year})</span>",
                   "font": {"size": 14, "color": "#166534", "family": "Plus Jakarta Sans"}},
            gauge={
                "axis": {"range": [0, max(pred_ton_ha * 2.5, 20)],
                         "tickwidth": 1, "tickcolor": "#cbd5e1",
                         "tickfont": {"color": "#64748b", "size": 11}},
                "bar": {"color": "#16a34a", "thickness": 0.28},
                "bgcolor": "#f8faf9", "borderwidth": 1, "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, pred_ton_ha * 0.4],  "color": "#fee2e2"},
                    {"range": [pred_ton_ha * 0.4, pred_ton_ha * 0.75], "color": "#fef3c7"},
                    {"range": [pred_ton_ha * 0.75, pred_ton_ha * 1.25], "color": "#dcfce7"},
                    {"range": [pred_ton_ha * 1.25, max(pred_ton_ha * 2.5, 20)], "color": "#e0f2fe"},
                ],
                "threshold": {"line": {"color": "#d97706", "width": 3},
                              "thickness": 0.75, "value": baseline_ton},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            height=280, margin=dict(l=25, r=25, t=55, b=10),
            font={"family": "Plus Jakarta Sans"},
        )
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Comparison bar chart
        fig_bar = go.Figure()
        for lbl, val, color in [
            ("Baseline Historis", baseline_ton, "#0284c7"),
            ("Prediksi Model", pred_ton_ha, "#16a34a"),
        ]:
            fig_bar.add_trace(go.Bar(
                x=[lbl], y=[val], marker_color=color, marker_line_width=0,
                text=[f"{val:,.2f} t/ha"], textposition="outside",
                textfont={"color": "#0f172a", "size": 12, "family": "Plus Jakarta Sans"},
                width=0.38,
            ))
        fig_bar.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8faf9",
            showlegend=False, height=240, margin=dict(l=10, r=10, t=15, b=10),
            yaxis={"gridcolor": "#e2ece5",
                   "tickfont": {"color": "#64748b", "size": 11},
                   "title": {"text": "ton/ha", "font": {"color": "#64748b", "size": 11}}},
            xaxis={"tickfont": {"color": "#1e293b", "size": 12}},
            font={"family": "Plus Jakarta Sans"},
        )
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Sensitivity Analysis ──────────────────────────────────────────────────
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>📈 Analisis Sensitivitas Agroklimat — Pengaruh Variabel terhadap Prediksi</div>",
        unsafe_allow_html=True,
    )

    def sweep(param, values, fixed):
        results = []
        for v in values:
            p = dict(fixed)
            p[param] = v
            hg, _, _ = predict_yield(p["area"], p["item"], int(p["year"]),
                                     p["rainfall"], p["pesticides"], p["avg_temp"])
            results.append(hg / 10_000)
        return results

    fixed = dict(area=area, item=item, year=year,
                 rainfall=rainfall, pesticides=pesticides, avg_temp=avg_temp)

    def line_chart(x_vals, y_vals, vline_val, xlabel, title, color, fillcolor):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines", fill="tozeroy",
            line=dict(color=color, width=2.5), fillcolor=fillcolor, name="Yield",
        ))
        fig.add_vline(x=vline_val, line_dash="dash", line_color="#d97706", line_width=2,
                      annotation_text=f"Input: {vline_val}", annotation_font_color="#b45309",
                      annotation_font_size=11)
        fig.update_layout(
            title=dict(text=title, font=dict(color="#166534", size=13), x=0),
            paper_bgcolor="#ffffff", plot_bgcolor="#f8faf9",
            height=240, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(title=xlabel, tickfont=dict(color="#64748b", size=11),
                       gridcolor="#e2ece5"),
            yaxis=dict(title="Yield (ton/ha)", tickfont=dict(color="#64748b", size=11),
                       gridcolor="#e2ece5"),
            showlegend=False, font=dict(family="Plus Jakarta Sans"),
        )
        return fig

    s1, s2 = st.columns(2, gap="large")
    with s1:
        rain_x = np.linspace(max(0, rainfall * 0.2), rainfall * 2.5, 40)
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(
            line_chart(rain_x, sweep("rainfall", rain_x, fixed), rainfall,
                       "Curah Hujan (mm/thn)", "🌧️ Respon terhadap Curah Hujan",
                       "#0284c7", "rgba(2, 132, 199, 0.08)"),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with s2:
        temp_x = np.linspace(max(-5, avg_temp - 15), avg_temp + 15, 40)
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(
            line_chart(temp_x, sweep("avg_temp", temp_x, fixed), avg_temp,
                       "Suhu (°C)", "🌡️ Respon terhadap Suhu Rata-rata",
                       "#ea580c", "rgba(234, 88, 12, 0.08)"),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown("</div>", unsafe_allow_html=True)

    s3, s4 = st.columns(2, gap="large")
    with s3:
        pest_x = np.linspace(0, max(pesticides * 3, 10000), 40)
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(
            line_chart(pest_x, sweep("pesticides", pest_x, fixed), pesticides,
                       "Pestisida (ton)", "🧪 Respon terhadap Penggunaan Pestisida",
                       "#7c3aed", "rgba(124, 58, 237, 0.08)"),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with s4:
        year_x = list(range(2010, 2051))
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(
            line_chart(year_x, sweep("year", year_x, fixed), year,
                       "Tahun", "📅 Proyeksi Tren Waktu (Tahun)",
                       "#16a34a", "rgba(22, 163, 74, 0.08)"),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DISCLAIMER & FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='info-badge'>"
    "🌾 <b>Catatan Agrikultur & Keterbatasan Model:</b> Model Random Forest dilatih menggunakan data historis "
    "global FAO/WorldBank (1990–2013). Model tidak memperhitungkan variabel mikro-agronomi lokal seperti varietas benih unggul, "
    "kandungan hara tanah (NPK), dan sistem irigasi teknis. Gunakan estimasi ini sebagai panduan proyeksi makro."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:0.8rem;margin-top:1.5rem;font-weight:500;'>"
    "🌱 Crop Yield Prediction Dashboard · Smart Agriculture AI · "
    "<code style='color:#166534;background:#dcfce7;padding:2px 6px;border-radius:4px;'>crop_yield_crispdm.ipynb</code></div>",
    unsafe_allow_html=True,
)
