"""
Dashboard Simulasi Prediksi Hasil Panen (Crop Yield Prediction)
Model: Random Forest (rf_model.pkl) — pipeline CRISP-DM
Author: Data Science Project - Syafiq

Halaman:
- Tab 1  "Overview Data & Model"  -> render_overview()
- Tab 2  "Simulasi Prediksi"       -> render_prediction()

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    "Maize": "Jagung",
    "Potatoes": "Kentang",
    "Rice, paddy": "Padi",
    "Sorghum": "Sorgum",
    "Soybeans": "Kedelai",
    "Wheat": "Gandum",
    "Cassava": "Singkong / Ubi Kayu",
    "Yams": "Ubi",
    "Sweet potatoes": "Ubi Jalar",
    "Plantains and others": "Pisang & Lainnya",
}
CROP_REVERSE = {v: k for k, v in CROP_TRANSLATION.items()}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS EKSTERNAL
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Harvest Planner — Prediksi Hasil Panen",
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
# HALAMAN 1 — OVERVIEW DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = ["#2E7D32", "#0288D1", "#7B1FA2", "#F59E0B", "#DC2626",
           "#0D9488", "#64748B", "#EA580C", "#4F46E5", "#84CC16"]
GREEN_BAR = dict(color="rgba(129, 199, 132, 0.85)", line=dict(color="#2E7D32", width=1.5))
BLUE_BAR  = dict(color="rgba(129, 212, 250, 0.85)", line=dict(color="#0288D1", width=1.5))
RED_BAR   = dict(color="rgba(252, 165, 165, 0.85)", line=dict(color="#DC2626", width=1.5))

# Hasil evaluasi dari notebook (data uji 2010–2013). Baseline & Random Forest
# juga dihitung ulang langsung di bawah; LR & XGBoost dari output notebook
# (xgboost tidak dipasang di dashboard agar dependensi tetap ringan).
MODEL_RESULTS = {
    "Baseline (Wilayah×Komoditas)": {"r2": 0.8781, "mae": 18581.48},
    "Regresi Linear":               {"r2": 0.9221, "mae": 14519.96},
    "XGBoost":                      {"r2": 0.9326, "mae": 12982.28},
    "Random Forest":                {"r2": 0.9513, "mae": 10517.10},
}

LABELS = {
    "Year": "Tahun",
    "yield_ton": "Hasil Panen (ton/ha)",
    "average_rain_fall_mm_per_year": "Curah Hujan (mm/tahun)",
    "pesticides_tonnes": "Pestisida (ton)",
    "avg_temp": "Suhu Rata-rata (°C)",
}


@st.cache_data(show_spinner=False)
def load_raw():
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["yield_ton"] = df["hg/ha_yield"] / 10_000
    df["Komoditas"] = df["Item"].map(CROP_TRANSLATION).fillna(df["Item"])
    return df


@st.cache_data(show_spinner="Mengevaluasi model pada data uji...")
def compute_model_eval():
    """Ulang evaluasi notebook: split by tahun (train ≤ 2009), baseline dari train saja,
    model memprediksi residual log-yield. Mengembalikan data uji + prediksi + importance."""
    mdl = load_model()
    df = load_raw().copy()
    df["log_yield"] = np.log1p(df["hg/ha_yield"])
    train, test = df[df["Year"] <= 2009], df[df["Year"] > 2009].copy()

    bmap = train.groupby(["Area", "Item"])["log_yield"].mean()
    gmean = train["log_yield"].mean()
    test["base_log"] = [bmap.get((a, i), gmean) for a, i in zip(test["Area"], test["Item"])]
    test["rainfall_temp"] = test["average_rain_fall_mm_per_year"] * test["avg_temp"]

    feat = list(mdl.feature_names_in_)
    items = pd.get_dummies(test["Item"], prefix="item").reindex(
        columns=[c for c in feat if c.startswith("item_")], fill_value=0)
    X = pd.concat([test[["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes",
                         "avg_temp", "rainfall_temp"]], items], axis=1)[feat]

    test["pred_ton"] = np.clip(np.expm1(test["base_log"].values + mdl.predict(X)), 0, None) / 10_000
    test["base_ton"] = np.clip(np.expm1(test["base_log"].values), 0, None) / 10_000
    test["error_ton"] = test["pred_ton"] - test["yield_ton"]
    test["abs_error"] = test["error_ton"].abs()

    # importance: gabungkan semua dummy komoditas jadi satu
    imp = pd.Series(mdl.feature_importances_, index=feat)
    item_imp = imp[[c for c in feat if c.startswith("item_")]].sum()
    imp = imp[[c for c in feat if not c.startswith("item_")]]
    imp.index = ["Curah hujan" if "rain" in i and "temp" not in i else
                 "Interaksi hujan × suhu" if i == "rainfall_temp" else
                 "Pestisida" if "pest" in i else
                 "Suhu" if i == "avg_temp" else "Tahun" for i in imp.index]
    imp["Jenis komoditas"] = item_imp
    return test, imp.sort_values()


def style_fig(fig, height=360, title=None, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=10, r=20, t=50 if title else 20, b=10),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Outfit", color="#334155"),
        title=dict(text=title, font=dict(size=15, color="#0F172A"), x=0.01) if title else None,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)
    return fig


def show(fig):
    # width="stretch" untuk Streamlit baru; fallback untuk versi lama
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def caption(text):
    st.markdown(f"<div class='chart-caption'>{text}</div>", unsafe_allow_html=True)


def section(num, icon, title):
    st.markdown(render_template(TPL_PATH, "section_title", number=num, icon=icon, title=title),
                unsafe_allow_html=True)


def hbar(series, title, bar_style, fmt="{:.2f}", xlabel="Rata-rata Hasil Panen (ton/ha)", height=360):
    s = series.sort_values()
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h", marker=bar_style,
        text=[fmt.format(v) for v in s.values], textposition="outside",
        textfont=dict(size=11, color="#166534"), cliponaxis=False,
    ))
    style_fig(fig, height=height, title=title, legend=False)
    fig.update_layout(margin=dict(l=10, r=50, t=50, b=10))
    fig.update_xaxes(title=xlabel, range=[0, s.max() * 1.2])
    return fig


def render_overview():
    df_all = load_raw()
    test, importance = compute_model_eval()

    # ── Filter global ────────────────────────────────────────────────────────
    f1, f2 = st.columns([1, 2])
    with f1:
        pilihan = ["Semua Komoditas"] + sorted(df_all["Komoditas"].unique())
        komo = st.selectbox("Filter komoditas", pilihan)
    with f2:
        y_min, y_max = int(df_all["Year"].min()), int(df_all["Year"].max())
        yr = st.slider("Rentang tahun", y_min, y_max, (y_min, y_max))

    df = df_all[df_all["Year"].between(*yr)]
    if komo != "Semua Komoditas":
        df = df[df["Komoditas"] == komo]
    if df.empty:
        st.warning("Tidak ada data untuk filter ini.")
        return

    # ── 1. Ringkasan ─────────────────────────────────────────────────────────
    section("1", "📌", "Ringkasan Data")
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "green-box", "Jumlah Observasi", f"{len(df):,}"),
        (c2, "blue-box", "Jumlah Negara", f"{df['Area'].nunique()}"),
        (c3, "info-box", "Jumlah Komoditas", f"{df['Komoditas'].nunique()}"),
        (c4, "green-box", "Rata-rata Hasil Panen", f"{df['yield_ton'].mean():.2f} ton/ha"),
    ]
    for col, cls, title, val in cards:
        with col:
            st.markdown(render_template(TPL_PATH, "metric_card", box_class=cls, title=title, value=val),
                        unsafe_allow_html=True)

    by_item = df.groupby("Komoditas")["yield_ton"].mean().sort_values(ascending=False)
    by_area = df.groupby("Area")["yield_ton"].mean().sort_values(ascending=False)
    corr_y = df[["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "yield_ton"]] \
        .corr()["yield_ton"].drop("yield_ton")
    max_r = corr_y.abs().max()
    kekuatan = "sangat lemah" if max_r < 0.2 else "lemah" if max_r < 0.4 else "sedang"
    body = (f"Pada periode <b>{yr[0]}–{yr[1]}</b>, "
            + (f"komoditas dengan rata-rata hasil panen tertinggi adalah <b>{by_item.index[0]}</b> "
               f"(<b>{by_item.iloc[0]:.1f} ton/ha</b>), " if komo == "Semua Komoditas" else
               f"rata-rata hasil panen <b>{komo}</b> adalah <b>{by_item.iloc[0]:.1f} ton/ha</b>, ")
            + f"dan negara tertinggi adalah <b>{by_area.index[0]}</b> (<b>{by_area.iloc[0]:.1f} ton/ha</b>). "
            f"Korelasi linear faktor numerik (tahun, hujan, suhu, pestisida) dengan hasil panen tergolong "
            f"<b>{kekuatan}</b> (|r| maksimum = {max_r:.2f}), sehingga wilayah dan jenis tanaman "
            f"menjadi penentu utama produktivitas.")
    st.markdown(render_template(TPL_PATH, "insight_banner", title="💡 Ringkasan Insight", body=body),
                unsafe_allow_html=True)

    # ── 2. Insight data ──────────────────────────────────────────────────────
    section("2", "🔎", "Insight dari Data Historis")

    a, b = st.columns(2)
    with a:
        show(hbar(by_item, "Rata-rata hasil panen per komoditas", GREEN_BAR, height=380))
    with b:
        trend = df.groupby(["Year", "Komoditas"], as_index=False)["yield_ton"].mean()
        fig = px.line(trend, x="Year", y="yield_ton", color="Komoditas", color_discrete_sequence=PALETTE,
                      labels={"Year": "Tahun", "yield_ton": "ton/ha"})
        style_fig(fig, height=380, title="Tren rata-rata hasil panen per tahun")
        show(fig)
    caption("Kentang dan ubi jalar berada jauh di atas komoditas lain. Garis tren yang datar atau naik pelan "
            "menunjukkan peningkatan produktivitas jangka panjang yang tidak besar.")

    a, b = st.columns(2)
    with a:
        show(hbar(by_area.head(10), "10 negara dengan hasil panen tertinggi", GREEN_BAR, height=380))
    with b:
        show(hbar(by_area.tail(10), "10 negara dengan hasil panen terendah", RED_BAR, height=380))

    fig = px.choropleth(by_area.reset_index(), locations="Area", locationmode="country names",
                        color="yield_ton", color_continuous_scale="Greens",
                        labels={"yield_ton": "ton/ha", "Area": "Negara"})
    style_fig(fig, height=430, title="Peta rata-rata hasil panen per negara")
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                    bgcolor="rgba(0,0,0,0)", landcolor="#F1F5F9", showcountries=True, countrycolor="#E2E8F0")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar=dict(title="ton/ha", len=0.7))
    show(fig)
    caption("Negara yang tidak muncul di peta tidak ada dalam dataset (atau namanya tidak dikenali peta).")

    a, b = st.columns(2)
    with a:
        cols = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "yield_ton"]
        names = ["Tahun", "Curah Hujan", "Pestisida", "Suhu", "Hasil Panen"]
        cm = df[cols].corr()
        fig = px.imshow(cm.values, x=names, y=names, text_auto=".2f", zmin=-1, zmax=1,
                        color_continuous_scale="RdBu_r", aspect="auto")
        style_fig(fig, height=400, title="Korelasi antar variabel numerik", legend=False)
        fig.update_layout(coloraxis_showscale=False)
        show(fig)
    with b:
        var = st.radio("Variabel pada sumbu X", ["average_rain_fall_mm_per_year", "avg_temp", "pesticides_tonnes"],
                       format_func=lambda v: LABELS[v], horizontal=True)
        samp = df.sample(min(len(df), 2500), random_state=42)
        fig = px.scatter(samp, x=var, y="yield_ton", color="Komoditas", opacity=0.55,
                         color_discrete_sequence=PALETTE, hover_data=["Area", "Year"],
                         log_x=(var == "pesticides_tonnes"), labels=LABELS)
        style_fig(fig, height=350, title=f"{LABELS[var]} vs hasil panen")
        show(fig)
    caption("Korelasi mendekati nol bukan berarti iklim tidak berpengaruh, tetapi pengaruhnya bergantung pada "
            "komoditas dan lokasi. Pestisida cenderung berkorelasi dengan negara berproduktivitas tinggi, "
            "bukan penyebab langsung hasil panen (korelasi ≠ kausalitas).")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Sebelum transformasi (ton/ha)", "Setelah log1p"))
    fig.add_trace(go.Histogram(x=df["yield_ton"], nbinsx=50, marker_color="#81C784", name="Asli"), 1, 1)
    fig.add_trace(go.Histogram(x=np.log1p(df["hg/ha_yield"]), nbinsx=50, marker_color="#0288D1", name="Log"), 1, 2)
    style_fig(fig, height=320, title="Distribusi hasil panen", legend=False)
    show(fig)
    caption("Distribusi asli miring ke kanan; transformasi log membuatnya lebih simetris sehingga cocok sebagai "
            "target pemodelan.")

    # ── 3. Insight model ─────────────────────────────────────────────────────
    section("3", "🤖", "Insight Pemodelan (Random Forest)")
    r2_rf = 1 - ((test["yield_ton"] - test["pred_ton"]) ** 2).sum() / ((test["yield_ton"] - test["yield_ton"].mean()) ** 2).sum()
    r2_bs = 1 - ((test["yield_ton"] - test["base_ton"]) ** 2).sum() / ((test["yield_ton"] - test["yield_ton"].mean()) ** 2).sum()
    mae_rf = test["abs_error"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, cls, title, val in [
        (c1, "green-box", "R² Random Forest (data uji)", f"{r2_rf:.4f}"),
        (c2, "blue-box", "MAE Random Forest", f"{mae_rf:.2f} ton/ha"),
        (c3, "info-box", "R² Baseline", f"{r2_bs:.4f}"),
        (c4, "green-box", "Tambahan R² dari ML", f"+{r2_rf - r2_bs:.4f}"),
    ]:
        with col:
            st.markdown(render_template(TPL_PATH, "metric_card", box_class=cls, title=title, value=val),
                        unsafe_allow_html=True)

    names = list(MODEL_RESULTS)
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Bar(x=names, y=[MODEL_RESULTS[n]["r2"] for n in names], marker=GREEN_BAR,
                               text=[f"{MODEL_RESULTS[n]['r2']:.4f}" for n in names], textposition="outside",
                               cliponaxis=False))
        style_fig(fig, height=360, title="Perbandingan R² (lebih tinggi lebih baik)", legend=False)
        fig.update_yaxes(range=[0.8, 1.0])
        fig.update_xaxes(tickfont=dict(size=10))
        show(fig)
    with b:
        fig = go.Figure(go.Bar(x=names, y=[MODEL_RESULTS[n]["mae"] / 10_000 for n in names], marker=BLUE_BAR,
                               text=[f"{MODEL_RESULTS[n]['mae'] / 10_000:.2f}" for n in names],
                               textposition="outside", cliponaxis=False))
        style_fig(fig, height=360, title="Perbandingan MAE, ton/ha (lebih rendah lebih baik)", legend=False)
        fig.update_yaxes(range=[0, 2.3])
        fig.update_xaxes(tickfont=dict(size=10))
        show(fig)
    caption("Sumbu R² dipotong mulai 0,80 agar selisih antar model terlihat. Baseline sederhana "
            "(rata-rata historis per wilayah×komoditas) sudah menjelaskan ±88% variasi; machine learning "
            "menambah sekitar 7 poin persentase.")

    a, b = st.columns(2)
    with a:
        samp = test.sample(min(len(test), 2000), random_state=42)
        lim = float(max(samp["yield_ton"].max(), samp["pred_ton"].max()))
        fig = px.scatter(samp, x="yield_ton", y="pred_ton", opacity=0.45, hover_data=["Area", "Komoditas", "Year"],
                         labels={"yield_ton": "Aktual (ton/ha)", "pred_ton": "Prediksi (ton/ha)"},
                         color_discrete_sequence=["#2E7D32"])
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(dash="dash", color="#DC2626"),
                                 name="Prediksi = Aktual"))
        style_fig(fig, height=380, title="Aktual vs prediksi (data uji 2010–2013)", legend=False)
        show(fig)
    with b:
        fig = go.Figure(go.Histogram(x=test["error_ton"].clip(-8, 8), nbinsx=60, marker_color="#81C784"))
        style_fig(fig, height=380, title="Distribusi galat (prediksi − aktual, ton/ha)", legend=False)
        fig.add_vline(x=0, line_dash="dash", line_color="#DC2626")
        show(fig)
    caption("Titik di dekat garis putus-putus berarti prediksi akurat. Galat terpusat di sekitar nol; "
            "galat dipotong di ±8 ton/ha pada histogram agar mudah dibaca.")

    a, b = st.columns(2)
    with a:
        err_item = test.groupby("Komoditas")["abs_error"].mean()
        show(hbar(err_item, "MAE per komoditas (ton/ha)", BLUE_BAR, xlabel="MAE (ton/ha)", height=380))
    with b:
        cnt = test.groupby("Area")["abs_error"].agg(["mean", "count"])
        err_area = cnt[cnt["count"] >= 20]["mean"].sort_values(ascending=False).head(10)
        show(hbar(err_area, "10 negara dengan galat terbesar (ton/ha)", RED_BAR, xlabel="MAE (ton/ha)", height=380))
    caption("Galat mutlak cenderung lebih besar pada komoditas dengan hasil panen tinggi (mis. kentang). "
            "Gunakan grafik ini untuk melihat di mana prediksi perlu diperlakukan lebih hati-hati.")

    fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h", marker=GREEN_BAR,
                           text=[f"{v:.1%}" for v in importance.values], textposition="outside", cliponaxis=False))
    style_fig(fig, height=330, title="Feature importance Random Forest (pada residual)", legend=False)
    fig.update_layout(margin=dict(l=10, r=60, t=50, b=10))
    fig.update_xaxes(range=[0, importance.max() * 1.2], title="Kontribusi relatif")
    show(fig)
    caption("Random Forest memodelkan <b>residual</b> di atas baseline wilayah×komoditas. Jadi grafik ini menunjukkan "
            "faktor yang menjelaskan sisa variasi, bukan seluruh variasi hasil panen.")

    # ── 4. Eksplorasi interaktif ─────────────────────────────────────────────
    section("4", "🧭", "Eksplorasi Negara & Komoditas")
    e1, e2 = st.columns(2)
    with e1:
        areas = sorted(df_all["Area"].unique())
        area = st.selectbox("Negara", areas, index=areas.index("Indonesia") if "Indonesia" in areas else 0,
                            key="ov_area")
    with e2:
        opts = sorted(df_all[df_all["Area"] == area]["Komoditas"].unique())
        komo_sel = st.selectbox("Komoditas", opts, index=opts.index("Padi") if "Padi" in opts else 0,
                                key="ov_komo")
    item_raw = CROP_REVERSE.get(komo_sel, komo_sel)

    hist = df_all[(df_all["Area"] == area) & (df_all["Item"] == item_raw)].sort_values("Year")
    pred = test[(test["Area"] == area) & (test["Item"] == item_raw)].sort_values("Year")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Year"], y=hist["yield_ton"], mode="lines+markers", name="Aktual",
                             line=dict(color="#2E7D32", width=2.5)))
    if not pred.empty:
        fig.add_trace(go.Scatter(x=pred["Year"], y=pred["pred_ton"], mode="lines+markers", name="Prediksi model (data uji)",
                                 line=dict(color="#DC2626", dash="dash", width=2)))
        fig.add_vrect(x0=2009.5, x1=2013.5, fillcolor="#FEF3C7", opacity=0.4, line_width=0,
                      annotation_text="Periode uji", annotation_position="top left")
    style_fig(fig, height=360, title=f"Hasil panen {komo_sel} di {area}: aktual vs prediksi")
    fig.update_yaxes(title="ton/ha")
    show(fig)

    last = hist.iloc[-1]
    rain0, temp0, pest0, yr0 = (float(last["average_rain_fall_mm_per_year"]), float(last["avg_temp"]),
                                float(last["pesticides_tonnes"]), int(last["Year"]))
    rain_pct = np.arange(-50, 51, 10)
    temp_d = np.arange(-3, 3.1, 0.5)
    y_rain = [predict_single(area, item_raw, yr0, rain0 * (1 + p / 100), temp0, pest0)[0] for p in rain_pct]
    y_temp = [predict_single(area, item_raw, yr0, rain0, temp0 + d, pest0)[0] for d in temp_d]

    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Scatter(x=rain_pct, y=y_rain, mode="lines+markers", line=dict(color="#0288D1", width=2.5)))
        style_fig(fig, height=330, title="Sensitivitas terhadap curah hujan", legend=False)
        fig.update_xaxes(title="Perubahan curah hujan (%)")
        fig.update_yaxes(title="Prediksi (ton/ha)")
        show(fig)
    with b:
        fig = go.Figure(go.Scatter(x=temp_d, y=y_temp, mode="lines+markers", line=dict(color="#DC2626", width=2.5)))
        style_fig(fig, height=330, title="Sensitivitas terhadap suhu", legend=False)
        fig.update_xaxes(title="Perubahan suhu (°C)")
        fig.update_yaxes(title="Prediksi (ton/ha)")
        show(fig)
    caption(f"Skenario what-if memakai kondisi tahun {yr0} ({rain0:,.0f} mm, {temp0:.1f}°C, {pest0:,.0f} ton pestisida) "
            f"dengan satu variabel diubah. Kurva yang hampir datar berarti model tidak sensitif terhadap variabel itu "
            f"untuk kombinasi ini.")

    # ── Catatan ──────────────────────────────────────────────────────────────
    st.markdown(render_template(
        TPL_PATH, "note_box", title="⚠️ Catatan & Keterbatasan",
        body=("Data hanya mencakup 1990–2013 dan curah hujan serta suhu adalah rata-rata tahunan per negara, "
              "sehingga variasi antarwilayah di dalam negara tidak tertangkap. Model dievaluasi dengan split berdasarkan "
              "waktu (latih 1990–2009, uji 2010–2013). Karena Random Forest tidak mengekstrapolasi, prediksi untuk tahun "
              "proyeksi jauh (mis. 2027–2045) praktis mengikuti pola tahun terakhir data, bukan tren baru. "
              "Hubungan yang tampak bersifat asosiatif, bukan sebab-akibat."),
    ), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER (KUSTOM HTML & CSS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    render_template(TPL_PATH, "hero"),
    unsafe_allow_html=True
)


def render_prediction():
    """Halaman 2 — Simulasi prediksi (kode asli, dibungkus fungsi)."""
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
            years_list = list(range(2027, 2046))
            year = st.selectbox("3. Tahun Proyeksi:", options=years_list, index=0)
            rain = st.number_input("4. Curah Hujan (mm/tahun)", value=1485.0, step=10.0)
            temp = st.number_input("5. Suhu Rata-rata (°C)", value=16.37, step=0.1, format="%.2f")
            pesticide = st.number_input("6. Penggunaan Pestisida (Ton)", value=121.0, step=1.0)

            submitted = st.form_submit_button("Jalankan Prediksi", use_container_width=True)

    item_raw = CROP_REVERSE.get(item_indo, item_indo)

    # Tandai bahwa form sudah pernah disubmit menggunakan session_state
    if submitted:
        st.session_state["prediction_done"] = True
        st.session_state["pred_params"] = {
            "area": area, "item_raw": item_raw, "item_indo": item_indo,
            "year": year, "rain": rain, "temp": temp, "pesticide": pesticide,
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # TAMPILKAN HASIL HANYA JIKA PREDIKSI SUDAH PERNAH DIJALANKAN
    # ─────────────────────────────────────────────────────────────────────────────
    if st.session_state.get("prediction_done"):
        p = st.session_state["pred_params"]
        area      = p["area"]
        item_raw  = p["item_raw"]
        year      = p["year"]
        rain      = p["rain"]
        temp      = p["temp"]
        pesticide = p["pesticide"]

        # ── OUTPUT 1 — PREDIKSI UTAMA ────────────────────────────────────────────
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

        # ── OUTPUT 2 — ANALISIS SENSITIVITAS ─────────────────────────────────────
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

        # ── OUTPUT 3 — PERANGKINGAN KOMODITAS & DSS ───────────────────────────────
        st.markdown(
            render_template(TPL_PATH, "section_title", number="3", icon="🏆",
                             title="Perangkingan Komoditas & Decision Support System (DSS)"),
            unsafe_allow_html=True,
        )

        rankings = []
        for raw_c in raw_item_list:
            ton_c, _ = predict_single(area, raw_c, year, rain, temp, pesticide)
            rankings.append({"crop": CROP_TRANSLATION.get(raw_c, raw_c), "yield_ton": round(ton_c, 2)})
        rankings.sort(key=lambda x: x["yield_ton"], reverse=True)
        top_crop, top_yield = rankings[0]["crop"], rankings[0]["yield_ton"]

        st.markdown(render_template(
            TPL_PATH, "dss_banner",
            temp=f"{temp:.2f}", rain=f"{rain:,.0f}", area=area,
            top_crop=top_crop, top_yield=f"{top_yield:,.2f}",
        ), unsafe_allow_html=True)

        rank_df = pd.DataFrame(rankings).sort_values("yield_ton")  # ascending, biar terbesar di atas saat horizontal
        max_val = rank_df["yield_ton"].max()
        fig = go.Figure(go.Bar(
            x=rank_df["yield_ton"], y=rank_df["crop"], orientation="h",
            marker=dict(color="rgba(129, 199, 132, 0.85)", line=dict(color="#2E7D32", width=1.5)),
            text=rank_df["yield_ton"].apply(lambda v: f"{v:,.2f} ton/ha"),
            textposition="outside",
            textfont=dict(size=11, color="#166534", family="Outfit"),
            cliponaxis=False,
        ))
        fig.update_layout(
            height=380, margin=dict(l=10, r=80, t=10, b=10),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            xaxis=dict(
                title="Hasil Panen (ton/ha)",
                gridcolor="#e2e8f0",
                range=[0, max_val * 1.25],  # beri ruang untuk label di luar bar
            ),
            yaxis=dict(tickfont=dict(size=12)),
            font=dict(family="Outfit"),
        )
        show(fig)

    else:
        # Pesan awal sebelum prediksi dijalankan
        st.markdown(
            """
            <div style="
                margin-top: 2.5rem;
                padding: 2.5rem;
                background: linear-gradient(135deg, #f0fdf4, #dcfce7);
                border: 1.5px dashed #86efac;
                border-radius: 16px;
                text-align: center;
                color: #166534;
            ">
                <div style="font-size: 3rem; margin-bottom: 0.75rem;">🌾</div>
                <h3 style="margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 700;">
                    Siap Memprediksi Hasil Panen!
                </h3>
                <p style="margin: 0; font-size: 0.97rem; opacity: 0.85;">
                    Isi parameter agroklimat di atas, lalu klik <strong>Jalankan Prediksi</strong>
                    untuk melihat estimasi hasil panen, analisis sensitivitas, dan perangkingan komoditas.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGASI HALAMAN
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_prediksi = st.tabs(["📊 Overview Data & Model", "🌾 Simulasi Prediksi"])
with tab_overview:
    render_overview()
with tab_prediksi:
    render_prediction()
