import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.predict import predict_modal

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Modal Usaha 💰", layout="wide", page_icon="💼")

# Define blue color palette
PRIMARY_BLUE = "#1E90FF"  # Dodger Blue
SECONDARY_BLUE = "#187bcd" # darker accent
LIGHT_BLUE = "#91B4D3"   # very light blue background
TEXT_COLOR = "#0f1724"
CARD_BG = "#f7fbff"

# === Styling CSS ===
st.markdown(
    f"""
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(180deg, {LIGHT_BLUE}, #f0f8ff);
        padding: 1.1rem;
    }}
    /* Sidebar title */
    .sidebar-heading {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 10px;
    }}
    .sidebar-heading img {{
        border-radius: 8px;
    }}
    /* Main header */
    .app-header {{
        display:flex;
        align-items:center;
        gap:16px;
    }}
    .app-header h1 {{
        margin:0;
        color: {PRIMARY_BLUE};
    }}
    /* Card style */
    .stat-card {{
        background: {CARD_BG};
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# === LOAD MODEL DAN DATA (cached) ===
@st.cache_resource
def load_resources():
    model_path = os.path.join("model", "model.pkl")
    model_data = joblib.load(model_path)
    data_path = os.path.join("data", "modal_usaha.csv")
    df_local = pd.read_csv(data_path)
    return model_data, df_local

model_data, df = load_resources()
le_jenis = model_data["le_jenis"]
le_lokasi = model_data["le_lokasi"]

# === SIDEBAR ===
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-heading">
            <img src="https://img.icons8.com/fluency/48/briefcase.png" alt="logo"/>
            <div>
                <div style="font-weight:bold; color:{PRIMARY_BLUE};">Prediksi Modal Usaha</div>
                <div style="font-size:12px; color:{TEXT_COLOR};">Estimasi modal dan analisis UMKM</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=["Prediksi Modal Usaha", "Statistik Usaha"],
        icons=["calculator", "bar-chart-line"],
        default_index=0,
        styles={
            "container": {"padding": "0px"},
            "nav-link": {
                "font-size": "15px",
                "color": TEXT_COLOR,
                "padding": "8px 10px",
                "--hover-color": "#e6f2ff",
                "border-radius": "6px",
            },
            "nav-link-selected": {
                "background-color": PRIMARY_BLUE,
                "color": "white",
                "font-weight": "600",
                "border-radius": "6px",
            },
        },
    )

# === MAIN LAYOUT ===
st.markdown(
    """
    <div class="app-header">
        <h1>Prediksi Modal Usaha</h1>
        <div style="color:#6b7280; margin-left:8px;">— Estimasi modal & insight bisnis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick dataset stats
total_rows = len(df)
avg_modal = df["Modal"].mean()
median_modal = df["Modal"].median()
unique_jenis = df["Jenis Usaha"].nunique()

col_a, col_b, col_c, col_d = st.columns([1.5,1,1,1])
with col_a:
    st.markdown(f'<div class="stat-card"><strong>Total Record</strong><div style="font-size:20px; color:{PRIMARY_BLUE}">{total_rows}</div></div>', unsafe_allow_html=True)
with col_b:
    st.markdown(f'<div class="stat-card"><strong>Avg Modal</strong><div style="font-size:20px; color:{PRIMARY_BLUE}">Rp {avg_modal:,.0f}</div></div>', unsafe_allow_html=True)
with col_c:
    st.markdown(f'<div class="stat-card"><strong>Median Modal</strong><div style="font-size:20px; color:{PRIMARY_BLUE}">Rp {median_modal:,.0f}</div></div>', unsafe_allow_html=True)
with col_d:
    st.markdown(f'<div class="stat-card"><strong>Jenis Usaha</strong><div style="font-size:20px; color:{PRIMARY_BLUE}">{unique_jenis}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# === HALAMAN 1: PREDIKSI MODAL ===
if selected == "Prediksi Modal Usaha":
    st.subheader("Input data usaha untuk estimasi modal")

    with st.form("form_prediksi"):
        left, right = st.columns(2)
        with left:
            jenis_usaha = st.selectbox("Jenis Usaha", le_jenis.classes_)
            lokasi = st.selectbox("Lokasi", le_lokasi.classes_)
            karyawan = st.number_input("Jumlah Karyawan", min_value=1, step=1, value=3)
            harga_bahan = st.number_input("Total Harga Bahan Baku (Rp)", min_value=0, step=50000, value=2000000)
        with right:
            target_produksi = st.number_input("Target Produksi (unit)", min_value=1, step=1, value=300)
            omset = st.number_input("Perkiraan Omset (Rp)", min_value=0, step=50000, value=12000000)
            st.write("")  # spacing
            submitted = st.form_submit_button("Prediksi Sekarang", use_container_width=True)

    if submitted:
        prediksi = predict_modal(jenis_usaha, lokasi, karyawan, harga_bahan, target_produksi, omset)
        st.success(f"Estimasi Modal Usaha: Rp {prediksi:,.0f}")
        st.markdown("""
            <div style="font-size:13px; color:#475569; margin-top:6px;">
                Hasil di atas adalah estimasi berdasarkan model. Untuk hasil lebih akurat, tambahkan data historis dan fitur relevan.
            </div>
        """, unsafe_allow_html=True)

# === HALAMAN 2: STATISTIK USAHA ===
elif selected == "Statistik Usaha":
    st.subheader("Visualisasi dan korelasi fitur")

    tab1, tab2, tab3 = st.tabs(["Omset vs Modal", "Karyawan vs Modal", "Heatmap Korelasi"])

    with tab1:
        st.markdown("Hubungan Omset dan Modal Usaha")
        plt.style.use("seaborn-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x="Omset", y="Modal", hue="Jenis Usaha", palette="Blues", s=90, ax=ax)
        ax.set_title("Omset vs Modal", color=TEXT_COLOR)
        ax.set_xlabel("Omset (Rp)")
        ax.set_ylabel("Modal (Rp)")
        st.pyplot(fig)

    with tab2:
        st.markdown("Jumlah Karyawan dan Modal")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x="Karyawan", y="Modal", palette="Blues", ax=ax2)
        ax2.set_title("Distribusi Modal berdasarkan Jumlah Karyawan", color=TEXT_COLOR)
        st.pyplot(fig2)

    with tab3:
        st.markdown("Korelasi antar variabel numerik")
        corr = df.select_dtypes(include=np.number).corr()
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax3, cbar_kws={"shrink": .8})
        ax3.set_title("Heatmap Korelasi", color=TEXT_COLOR)
        st.pyplot(fig3)