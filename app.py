"""
Dashboard Prediksi Hasil Panen (Crop Yield Prediction)
Disusun mengikuti alur CRISP-DM dari notebook crop_yield_crispdm.ipynb
Author: Data Science Project - Syafiq
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(
    page_title="Crop Yield Prediction Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f7f0;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #4CAF50;
    }
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #1b4332;}
    .stTabs [data-baseweb="tab"] {font-size: 16px; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

TRAIN_CUTOFF = 2009

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "yield_df.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "notebook_metrics.json"
NOTEBOOK_PATH = BASE_DIR / "crop_yield_crispdm.ipynb"

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat dataset...")
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    required = {
        "Area", "Item", "Year", "hg/ha_yield",
        "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di file: {missing}")
    return df


# ----------------------------------------------------------------------------
# PREPROCESSING + FEATURE ENGINEERING (mengikuti notebook)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Menyiapkan data (preprocessing & feature engineering)...")
def prepare_data(df: pd.DataFrame):
    df = df.drop_duplicates().copy()
    df["log_yield"] = np.log1p(df["hg/ha_yield"])

    train_df = df[df["Year"] <= TRAIN_CUTOFF].copy()
    test_df = df[df["Year"] > TRAIN_CUTOFF].copy()

    # Baseline: rata-rata log_yield per Area+Item
    baseline_map = train_df.groupby(["Area", "Item"])["log_yield"].mean()
    global_mean = train_df["log_yield"].mean()

    def get_baseline(data):
        return data.apply(
            lambda row: baseline_map.get((row["Area"], row["Item"]), global_mean),
            axis=1,
        )

    train_df["baseline_log_pred"] = get_baseline(train_df)
    test_df["baseline_log_pred"] = get_baseline(test_df)

    train_df["residual"] = train_df["log_yield"] - train_df["baseline_log_pred"]
    test_df["residual"] = test_df["log_yield"] - test_df["baseline_log_pred"]

    FEATURES = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]

    for data in [train_df, test_df]:
        data["rainfall_temp"] = data["average_rain_fall_mm_per_year"] * data["avg_temp"]
        data["pesticide_per_year"] = data["pesticides_tonnes"] / (data["Year"] - 1989 + 1)

    FEATURES += ["rainfall_temp", "pesticide_per_year"]

    item_train = pd.get_dummies(train_df["Item"], prefix="item")
    item_test = pd.get_dummies(test_df["Item"], prefix="item")
    item_test = item_test.reindex(columns=item_train.columns, fill_value=0)

    X_train = pd.concat([train_df[FEATURES].reset_index(drop=True), item_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_df[FEATURES].reset_index(drop=True), item_test.reset_index(drop=True)], axis=1)
    y_train = train_df["residual"]
    y_test = test_df["residual"]

    return {
        "df": df, "train_df": train_df, "test_df": test_df,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "baseline_map": baseline_map, "global_mean": global_mean,
        "item_columns": item_train.columns, "features": FEATURES,
    }


# ----------------------------------------------------------------------------
# MODELLING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Melatih model (Linear Regression, Random Forest, XGBoost)...")
def train_models(_prep):
    X_train, y_train = _prep["X_train"], _prep["y_train"]

    linear_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_train, y_train)

    models = {"Linear Regression": linear_model, "Random Forest": rf_model}

    if HAS_XGB:
        xgb_model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42,
        ).fit(X_train, y_train)
        models["XGBoost"] = xgb_model

    return models


def to_yield(pred_residual, baseline_log_pred):
    pred_log = baseline_log_pred.values + pred_residual
    pred = np.expm1(pred_log)
    return np.clip(pred, 0, None)


@st.cache_data(show_spinner="Menghitung evaluasi model...")
def evaluate_models(_models, _prep):
    X_test = _prep["X_test"]
    test_df = _prep["test_df"]
    actual = test_df["hg/ha_yield"].values

    rows = []
    predictions = {}
    for name, model in _models.items():
        pred_res = model.predict(X_test)
        pred_yield = to_yield(pred_res, test_df["baseline_log_pred"])
        predictions[name] = pred_yield
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(actual, pred_yield),
            "RMSE": np.sqrt(mean_squared_error(actual, pred_yield)),
            "R2": r2_score(actual, pred_yield),
        })

    # Baseline
    baseline_pred = np.clip(np.expm1(test_df["baseline_log_pred"]), 0, None)
    rows.append({
        "Model": "Baseline (rata-rata Area+Item)",
        "MAE": mean_absolute_error(actual, baseline_pred),
        "RMSE": np.sqrt(mean_squared_error(actual, baseline_pred)),
        "R2": r2_score(actual, baseline_pred),
    })
    predictions["Baseline (rata-rata Area+Item)"] = baseline_pred

    results_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return results_df, predictions, actual


def predict_yield(model, prep, area, item, year, rainfall, pesticides, avg_temp):
    baseline = prep["baseline_map"].get((area, item), prep["global_mean"])

    input_data = pd.DataFrame({
        "Year": [year],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [avg_temp],
        "rainfall_temp": [rainfall * avg_temp],
        "pesticide_per_year": [pesticides / (year - 1989 + 1)],
    })

    item_input = pd.DataFrame(0, index=[0], columns=prep["item_columns"])
    item_col = f"item_{item}"
    if item_col in item_input.columns:
        item_input[item_col] = 1

    X_input = pd.concat([input_data, item_input], axis=1)
    X_input = X_input[prep["X_train"].columns]

    residual_pred = model.predict(X_input)[0]
    log_pred = baseline + residual_pred
    prediction = np.expm1(log_pred)
    return max(0, prediction), (baseline == prep["global_mean"])


# ----------------------------------------------------------------------------
# SIDEBAR: DATA SOURCE
# ----------------------------------------------------------------------------
st.sidebar.title("🌾 Crop Yield Dashboard")
st.sidebar.caption("Dataset: Kaggle - Crop Yield Prediction Dataset (`yield_df.csv`)")

data_source = None
if DATA_PATH.exists():
    data_source = DATA_PATH
    st.sidebar.success(f"✅ Dataset dimuat dari `data/{DATA_PATH.name}`")
else:
    st.sidebar.warning("⚠️ `data/yield_df.csv` tidak ditemukan di folder proyek.")
    uploaded_file = st.sidebar.file_uploader("Upload `yield_df.csv`", type=["csv"])
    if uploaded_file is not None:
        data_source = uploaded_file

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigasi",
    [
        "🏠 Business Understanding",
        "🔎 Data Understanding",
        "🛠️ Data Preparation",
        "🤖 Modelling & Evaluation",
        "🧩 Feature Importance",
        "📁 Hasil Mining (Notebook)",
        "🎯 Simulasi Prediksi",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dashboard ini mereplikasi alur CRISP-DM: Business Understanding → "
    "Data Understanding → Data Preparation → Modelling → Evaluation → "
    "Deployment (simulasi prediksi)."
)

if data_source is None:
    st.title("🌾 Crop Yield Prediction Dashboard")
    st.info(
        "Letakkan file **`yield_df.csv`** dari dataset "
        "[Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset/data) "
        "ke dalam folder **`data/`** proyek ini (atau upload lewat sidebar) untuk memulai."
    )
    st.markdown(
        """
        **Kolom yang dibutuhkan pada file CSV:**
        - `Area` — wilayah/negara
        - `Item` — jenis tanaman
        - `Year` — tahun
        - `hg/ha_yield` — hasil panen (hektogram per hektare)
        - `average_rain_fall_mm_per_year` — curah hujan rata-rata
        - `pesticides_tonnes` — penggunaan pestisida
        - `avg_temp` — suhu rata-rata
        """
    )
    st.stop()

try:
    df_raw = load_data(data_source)
except Exception as e:
    st.error(f"Gagal memuat file: {e}")
    st.stop()

prep = prepare_data(df_raw)
models = train_models(prep)
results_df, predictions, actual_yield = evaluate_models(models, prep)
best_model_name = results_df.iloc[0]["Model"]

# ============================================================================
# PAGE 1: BUSINESS UNDERSTANDING
# ============================================================================
if page == "🏠 Business Understanding":
    st.title("🏠 Business Understanding")

    st.markdown("### Latar Belakang")
    st.write(
        "Produktivitas pertanian dipengaruhi oleh berbagai faktor seperti kondisi iklim, "
        "curah hujan, suhu, penggunaan pestisida, jenis tanaman, dan wilayah. Prediksi hasil "
        "panen dapat membantu memberikan estimasi produktivitas sehingga dapat digunakan "
        "sebagai salah satu dasar dalam perencanaan produksi pertanian."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Tujuan")
        st.write(
            "Membangun model untuk memprediksi hasil panen (hg/ha) berdasarkan:\n"
            "- Wilayah (Area)\n- Jenis tanaman (Item)\n- Tahun\n"
            "- Curah hujan\n- Penggunaan pestisida\n- Suhu rata-rata"
        )
    with col2:
        st.markdown("### 📋 Tujuan Analisis")
        st.write(
            "1. Mengidentifikasi pola hasil panen.\n"
            "2. Mengetahui kontribusi faktor terhadap prediksi yield.\n"
            "3. Membangun model prediksi yang memiliki performa baik.\n"
            "4. Menghasilkan simulasi prediksi sebagai decision-support sederhana."
        )

    st.markdown("### ⚠️ Batasan")
    st.warning(
        "Dataset bersifat **observasional** sehingga hubungan yang ditemukan merupakan "
        "asosiasi dan tidak dapat langsung dianggap sebagai hubungan sebab-akibat. "
        "Dataset juga tidak memiliki fitur agronomis penting seperti jenis tanah, dosis "
        "pupuk per hektare, status irigasi, dan lama masa tanam — sehingga hasil prediksi "
        "sebaiknya digunakan sebagai estimasi awal, bukan keputusan final."
    )

    st.markdown("### 🗺️ Alur CRISP-DM Proyek")
    steps = [
        "1. Business Understanding", "2. Data Understanding", "3. Data Preparation",
        "4. Modelling", "5. Evaluation", "6. Interpretation", "7. Deployment / Simulation",
    ]
    cols = st.columns(len(steps))
    for c, s in zip(cols, steps):
        c.markdown(f"<div class='metric-card' style='text-align:center'>{s}</div>", unsafe_allow_html=True)

# ============================================================================
# PAGE 2: DATA UNDERSTANDING
# ============================================================================
elif page == "🔎 Data Understanding":
    st.title("🔎 Data Understanding")

    df = prep["df"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Baris", f"{df.shape[0]:,}")
    c2.metric("Jumlah Kolom", f"{df.shape[1]}")
    c3.metric("Jumlah Wilayah", f"{df['Area'].nunique()}")
    c4.metric("Jumlah Jenis Tanaman", f"{df['Item'].nunique()}")

    st.markdown("#### Cuplikan Data")
    st.dataframe(df.head(10), use_container_width=True)

    qc1, qc2 = st.columns(2)
    with qc1:
        st.markdown("#### ✅ Missing Value")
        st.dataframe(df.isnull().sum().rename("Jumlah Missing").to_frame(), use_container_width=True)
    with qc2:
        st.markdown("#### ✅ Data Duplikat")
        st.metric("Jumlah baris duplikat", f"{df_raw.duplicated().sum()}")
        st.caption("Data lengkap tanpa missing value pada notebook sumber.")

    st.markdown("---")
    st.markdown("### 📊 Distribusi Hasil Panen")
    fig = px.histogram(
        df, x=(df["hg/ha_yield"] / 10000), nbins=50, marginal="box",
        labels={"x": "Hasil Panen (ton/ha)"}, title="Distribusi Hasil Panen (ton/ha)",
    )
    fig.update_layout(xaxis_title="Hasil Panen (ton/ha)", yaxis_title="Jumlah Data")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Sebagian besar data memiliki hasil panen di bawah ~10 ton/ha, dengan sebagian "
        "kecil data mencapai hingga ~50 ton/ha — distribusi menjulur ke kanan (right-skewed)."
    )

    st.markdown("### 🌾 Rata-rata Hasil Panen per Jenis Tanaman")
    yield_item = (df.groupby("Item")["hg/ha_yield"].mean().sort_values() / 10000).reset_index()
    yield_item.columns = ["Item", "ton_per_ha"]
    fig2 = px.bar(
        yield_item, x="ton_per_ha", y="Item", orientation="h",
        title="Rata-rata Hasil Panen Berdasarkan Jenis Tanaman",
        labels={"ton_per_ha": "Rata-rata Hasil Panen (ton/ha)", "Item": ""},
        color="ton_per_ha", color_continuous_scale="Greens",
    )
    fig2.update_layout(height=max(400, 25 * len(yield_item)), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🌍 15 Wilayah dengan Rata-rata Hasil Panen Tertinggi")
    rata_area = (
        df.groupby("Area")["hg/ha_yield"].mean().sort_values(ascending=False).head(15) / 10000
    ).sort_values().reset_index()
    rata_area.columns = ["Area", "ton_per_ha"]
    fig3 = px.bar(
        rata_area, x="ton_per_ha", y="Area", orientation="h",
        title="15 Wilayah dengan Rata-rata Hasil Panen Tertinggi",
        labels={"ton_per_ha": "Rata-rata Hasil Panen (ton/ha)", "Area": ""},
        color="ton_per_ha", color_continuous_scale="Blues",
    )
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🔥 Korelasi Variabel Numerik")
    numeric_cols = ["Year", "hg/ha_yield", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    corr = df[numeric_cols].corr()
    fig4 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Korelasi Variabel Numerik",
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "Korelasi linear antar variabel numerik terhadap hasil panen tergolong lemah "
        "(curah hujan ≈0.00, pestisida ≈0.06, suhu ≈-0.11) — mengindikasikan hubungan "
        "yang tidak sepenuhnya linear, sehingga model non-linear seperti Random Forest "
        "berpotensi menangkap pola lebih baik."
    )

# ============================================================================
# PAGE 3: DATA PREPARATION
# ============================================================================
elif page == "🛠️ Data Preparation":
    st.title("🛠️ Data Preparation")

    st.markdown("### 1️⃣ Transformasi Log pada Target")
    st.write(
        "`hg/ha_yield` memiliki rentang nilai yang sangat lebar sehingga ditransformasi "
        "menjadi `log_yield = log1p(hg/ha_yield)` agar skala lebih terkendali untuk pemodelan."
    )
    df = prep["df"]
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=df["hg/ha_yield"] / 10000, nbins=30, title="Sebelum Transformasi")
        fig.update_layout(xaxis_title="Hasil Panen (ton/ha)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="log_yield", nbins=30, title="Setelah Transformasi Log")
        fig.update_layout(xaxis_title="Log Hasil Panen")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2️⃣ Split Data Berdasarkan Tahun")
    st.write(
        f"Data dibagi berdasarkan tahun agar evaluasi mendekati kondisi nyata (model belajar "
        f"dari masa lalu untuk memprediksi masa depan): **Train ≤ {TRAIN_CUTOFF}**, "
        f"**Test > {TRAIN_CUTOFF}**."
    )
    c1, c2 = st.columns(2)
    c1.metric("Data Train", f"{prep['train_df'].shape[0]:,} baris",
               f"{prep['train_df']['Year'].min()}–{prep['train_df']['Year'].max()}")
    c2.metric("Data Test", f"{prep['test_df'].shape[0]:,} baris",
               f"{prep['test_df']['Year'].min()}–{prep['test_df']['Year'].max()}")

    st.markdown("### 3️⃣ Baseline Prediction (Area + Item)")
    st.write(
        "Dibuat prediksi baseline sederhana berupa rata-rata `log_yield` per kombinasi "
        "`Area` + `Item`. Model machine learning nantinya memprediksi **residual** (selisih "
        "dari baseline ini), bukan nilai yield secara langsung — sehingga model fokus "
        "mempelajari koreksi dari baseline."
    )
    st.dataframe(prep["train_df"]["residual"].describe().to_frame("residual (train)"), use_container_width=True)

    st.markdown("### 4️⃣ Feature Engineering")
    st.write(
        "- `rainfall_temp` = curah hujan × suhu rata-rata (interaksi iklim)\n"
        "- `pesticide_per_year` = pestisida dibagi lama periode tahun sejak 1990\n"
        "- `Item` diubah menjadi one-hot encoding"
    )
    st.markdown(f"**Total fitur akhir:** {prep['X_train'].shape[1]} kolom "
                f"({prep['X_train'].shape[0]:,} baris train, {prep['X_test'].shape[0]:,} baris test)")
    st.dataframe(prep["X_train"].head(), use_container_width=True)

# ============================================================================
# PAGE 4: MODELLING & EVALUATION
# ============================================================================
elif page == "🤖 Modelling & Evaluation":
    st.title("🤖 Modelling & Evaluation")

    st.markdown(
        "Tiga algoritma dilatih pada data yang sama untuk memprediksi **residual** dari "
        "baseline: **Linear Regression**, **Random Forest**, dan **XGBoost**"
        + ("" if HAS_XGB else " *(XGBoost tidak tersedia di environment ini, dilewati)*") + "."
    )

    st.markdown("### 📈 Perbandingan Metrik Evaluasi")
    fmt_df = results_df.copy()
    fmt_df["MAE"] = fmt_df["MAE"].map(lambda v: f"{v:,.2f}")
    fmt_df["RMSE"] = fmt_df["RMSE"].map(lambda v: f"{v:,.2f}")
    fmt_df["R2"] = fmt_df["R2"].map(lambda v: f"{v:.4f}")
    st.dataframe(fmt_df, use_container_width=True, hide_index=True)

    best_row = results_df.iloc[0]
    st.success(
        f"🏆 **Model terbaik: {best_row['Model']}** — R² = {best_row['R2']:.4f} "
        f"({best_row['R2']*100:.2f}%), MAE = {best_row['MAE']:,.2f} hg/ha"
    )

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        fig = px.bar(
            results_df, x="Model", y="R2", title="Perbandingan R² Antar Model",
            color="R2", color_continuous_scale="Greens", text_auto=".4f",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with mcol2:
        fig = px.bar(
            results_df, x="Model", y="MAE", title="Perbandingan MAE Antar Model",
            color="MAE", color_continuous_scale="Reds", text_auto=".2s",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Actual vs Predicted")
    model_choice = st.selectbox(
        "Pilih model untuk divisualisasikan:",
        [m for m in results_df["Model"] if m != "Baseline (rata-rata Area+Item)"],
    )
    pred = predictions[model_choice]
    actual_ton = actual_yield / 10000
    pred_ton = pred / 10000

    scatter_df = pd.DataFrame({"Aktual (ton/ha)": actual_ton, "Prediksi (ton/ha)": pred_ton})
    fig = px.scatter(
        scatter_df, x="Aktual (ton/ha)", y="Prediksi (ton/ha)", opacity=0.4,
        title=f"Hasil Panen Aktual vs Prediksi — {model_choice}",
    )
    lim = [0, max(scatter_df.max()) * 1.05]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="red"), name="Prediksi Sempurna"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Semakin dekat titik-titik dengan garis putus-putus merah, semakin akurat prediksi model.")

# ============================================================================
# PAGE 5: FEATURE IMPORTANCE
# ============================================================================
elif page == "🧩 Feature Importance":
    st.title("🧩 Feature Importance")

    tree_models = {k: v for k, v in models.items() if hasattr(v, "feature_importances_")}
    if not tree_models:
        st.info("Tidak ada model berbasis tree yang tersedia untuk feature importance.")
    else:
        model_choice = st.selectbox("Pilih model:", list(tree_models.keys()))
        model = tree_models[model_choice]
        importance_df = pd.DataFrame({
            "Fitur": prep["X_train"].columns,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(15)

        fig = px.bar(
            importance_df.sort_values("Importance"), x="Importance", y="Fitur", orientation="h",
            title=f"Top 15 Feature Importance — {model_choice}",
            color="Importance", color_continuous_scale="Viridis",
        )
        fig.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Feature importance menunjukkan seberapa besar kontribusi setiap fitur dalam "
            "membantu model mengoreksi prediksi baseline (residual). Pada notebook sumber, "
            "analisis SHAP menunjukkan `Year`, `pesticides_tonnes`, `pesticide_per_year`, "
            "jenis tanaman jagung (Maize), dan `rainfall_temp` sebagai fitur paling berpengaruh."
        )

# ============================================================================
# PAGE 6: HASIL MINING DARI NOTEBOOK (arsip)
# ============================================================================
elif page == "📁 Hasil Mining (Notebook)":
    st.title("📁 Hasil Mining Data (dari Notebook Asli)")
    st.write(
        "Halaman ini menampilkan **output asli** dari `crop_yield_crispdm.ipynb` yang sudah "
        "dijalankan sebelumnya (gambar & metrik tersimpan di folder `results/`) — sebagai arsip "
        "hasil eksperimen, berdampingan dengan hasil training *live* di halaman lain."
    )

    if NOTEBOOK_PATH.exists():
        with open(NOTEBOOK_PATH, "rb") as f:
            st.download_button(
                "⬇️ Download Notebook Asli (crop_yield_crispdm.ipynb)",
                data=f,
                file_name="crop_yield_crispdm.ipynb",
                mime="application/x-ipynb+json",
                use_container_width=True,
            )
    else:
        st.info("File `crop_yield_crispdm.ipynb` tidak ditemukan di folder proyek.")

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            nb_metrics = json.load(f)

        st.markdown("### 📊 Metrik Evaluasi (hasil run notebook)")
        nb_results_df = pd.DataFrame(nb_metrics["results"])
        nb_results_df.columns = ["Model", "MAE", "RMSE", "R2"]
        st.dataframe(nb_results_df, use_container_width=True, hide_index=True)
        st.success(f"🏆 Model terbaik pada notebook: **{nb_metrics['best_model']}**")

        with st.expander("🔍 Contoh simulasi prediksi dari notebook"):
            ex = nb_metrics["example_prediction"]
            st.json(ex)

        st.caption(nb_metrics.get("notes", ""))
    else:
        st.warning("File `results/notebook_metrics.json` tidak ditemukan.")

    st.markdown("---")
    st.markdown("### 🖼️ Visualisasi Asli dari Notebook")

    figure_captions = {
        "01_distribusi_yield.png": "Distribusi Hasil Panen (sebelum transformasi)",
        "02_rata_rata_per_item.png": "Rata-rata Hasil Panen per Jenis Tanaman",
        "03_top15_wilayah.png": "15 Wilayah dengan Rata-rata Hasil Panen Tertinggi",
        "04_korelasi_numerik.png": "Korelasi Variabel Numerik",
        "05_before_after_log.png": "Perbandingan Sebelum vs Sesudah Transformasi Log",
        "06_actual_vs_predicted_rf.png": "Actual vs Predicted — Random Forest",
    }

    if FIGURES_DIR.exists():
        images = sorted(FIGURES_DIR.glob("*.png"))
        if images:
            cols = st.columns(2)
            for idx, img_path in enumerate(images):
                caption = figure_captions.get(img_path.name, img_path.stem)
                with cols[idx % 2]:
                    st.image(str(img_path), caption=caption, use_container_width=True)
        else:
            st.info("Belum ada gambar tersimpan di `results/figures/`.")
    else:
        st.info("Folder `results/figures/` tidak ditemukan.")

# ============================================================================
# PAGE 7: SIMULASI PREDIKSI
# ============================================================================
elif page == "🎯 Simulasi Prediksi":
    st.title("🎯 Simulasi Prediksi Hasil Panen")
    st.write(
        "Masukkan kondisi wilayah, tanaman, dan iklim untuk mendapatkan estimasi hasil panen. "
        "Model memprediksi **residual** terhadap baseline (rata-rata historis Area+Item), lalu "
        "hasilnya dikembalikan ke skala hg/ha."
    )

    df = prep["df"]
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            area = st.selectbox("Wilayah (Area)", sorted(df["Area"].unique()))
            item = st.selectbox("Jenis Tanaman (Item)", sorted(df["Item"].unique()))
        with c2:
            year = st.number_input("Tahun", min_value=1990, max_value=2100, value=2024, step=1)
            avg_temp = st.number_input("Suhu Rata-rata (°C)", value=20.0, step=0.1, format="%.1f")
        with c3:
            rainfall = st.number_input("Curah Hujan (mm/tahun)", min_value=0.0, value=1500.0, step=10.0)
            pesticides = st.number_input("Penggunaan Pestisida (ton)", min_value=0.0, value=100.0, step=1.0)

        model_choice = st.selectbox("Model untuk prediksi:", list(models.keys()), index=list(models.keys()).index(best_model_name) if best_model_name in models else 0)
        submitted = st.form_submit_button("🔮 Prediksi Hasil Panen", use_container_width=True)

    if submitted:
        model = models[model_choice]
        pred_hg, used_global_mean = predict_yield(model, prep, area, item, int(year), rainfall, pesticides, avg_temp)
        pred_ton = pred_hg / 10000

        st.markdown("### 📊 Hasil Prediksi")
        r1, r2 = st.columns(2)
        r1.metric("Prediksi Hasil Panen", f"{pred_hg:,.2f} hg/ha")
        r2.metric("Setara", f"{pred_ton:,.2f} ton/ha")

        if used_global_mean:
            st.warning(
                f"Kombinasi **{area} + {item}** tidak ditemukan pada data historis (train). "
                "Prediksi menggunakan rata-rata global sebagai baseline, sehingga akurasinya "
                "bisa lebih rendah dibanding kombinasi yang sudah pernah terlihat oleh model."
            )
        else:
            hist_mean = prep["baseline_map"].get((area, item))
            hist_ton = np.expm1(hist_mean) / 10000
            st.caption(f"Rata-rata historis {area} + {item} pada data train: ~{hist_ton:,.2f} ton/ha")

    st.markdown("---")
    st.markdown("### ⚠️ Catatan Keterbatasan Model")
    st.info(
        "Dataset ini tidak memiliki fitur agronomis penting seperti **jenis tanah, dosis "
        "pupuk per hektare, status irigasi, dan lama masa tanam**. Hasil prediksi di atas "
        "adalah estimasi berbasis pola historis (Area, Item, Tahun, Curah Hujan, Pestisida, "
        "Suhu) dan sebaiknya digunakan sebagai referensi awal, bukan keputusan final."
    )

st.markdown("---")
st.caption("Dashboard dibuat berdasarkan notebook `crop_yield_crispdm.ipynb` — Crop Yield Prediction Project (CRISP-DM).")
