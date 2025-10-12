import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict import predict_modal

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Modal Usaha 💰", layout="wide", page_icon="💼")

# === LOAD MODEL DAN DATA ===
model_path = os.path.join("model", "model.pkl")
model_data = joblib.load(model_path)
le_jenis = model_data["le_jenis"]
le_lokasi = model_data["le_lokasi"]

data_path = os.path.join("data", "modal_usaha.csv")
df = pd.read_csv(data_path)

# === SIDEBAR ===
with st.sidebar:
    selected = option_menu(
        menu_title="Dashboard",
        options=["Statistik Usaha", "Prediksi Modal Usaha"],
        icons=["calculator", "bar-chart-line"],
        # menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#f8f9fa"},
            "icon": {"color": "#ff9800", "font-size": "20px"},
            "nav-link": {
                "font-size": "15px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#ffd54f"},
        },
    )

# === HALAMAN 1: PREDIKSI MODAL ===
if selected == "Prediksi Modal Usaha":
    st.title("Prediksi Modal Usaha")
    st.markdown(
        "Masukkan data di bawah ini untuk memprediksi **modal usaha** yang dibutuhkan."
    )

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            jenis_usaha = st.selectbox(
                "Jenis Usaha",
                le_jenis.classes_,
                help="Pilih jenis usaha seperti Kuliner, Fashion, Elektronik, dll.",
            )
            lokasi = st.selectbox(
                "Lokasi", le_lokasi.classes_, help="Pilih kota tempat usaha beroperasi."
            )
            karyawan = st.number_input(
                "Jumlah Karyawan",
                min_value=1,
                step=1,
                help="Total jumlah pegawai di usaha Anda.",
            )

        with col2:
            harga_bahan = st.number_input(
                "Total Harga Bahan Baku(Rp)",
                min_value=0,
                step=50000,
                help="Total biaya bahan baku utama yang dibutuhkan untuk produksi.",
            )
            target_produksi = st.number_input(
                "Target Produksi (unit)",
                min_value=1,
                step=10,
                help="Jumlah barang yang ditargetkan diproduksi per bulan.",
            )
            omset = st.number_input(
                "Perkiraan Omset (Rp)",
                min_value=0,
                step=50000,
                help="Perkiraan total pendapatan (penjualan).",
            )

        submitted = st.form_submit_button("🔮 Prediksi Modal")

    if submitted:
        prediksi = predict_modal(
            jenis_usaha, lokasi, karyawan, harga_bahan, target_produksi, omset
        )
        st.success(f"💸 Estimasi Modal Usaha: **Rp {prediksi:,.0f}**")

# === HALAMAN 2: STATISTIK USAHA ===
elif selected == "Statistik Usaha":
    st.title("Statistik & Analisis Data Usaha")
    st.markdown("Visualisasi hubungan antara berbagai variabel dalam dataset.")

    tab1, tab2, tab3 = st.tabs(
        ["Omset vs Modal", "Karyawan vs Modal", "Heatmap Korelasi"]
    )

    with tab1:
        st.markdown("### Hubungan Omset dan Modal Usaha")
        fig, ax = plt.subplots(figsize=(15, 5)) 
        sns.scatterplot(
            data=df,
            x="Omset",
            y="Modal",
            hue="Jenis Usaha",
            style="Lokasi",
            s=80, 
        )
        plt.title("Hubungan Omset dan Modal Usaha", fontsize=12)
        plt.xlabel("Omset (Rp)", fontsize=10)
        plt.ylabel("Modal (Rp)", fontsize=10)
        st.pyplot(fig)

    with tab2:
        st.markdown("### Hubungan Jumlah Karyawan dan Modal Usaha")
        fig2, ax2 = plt.subplots(figsize=(15, 5)) 
        sns.scatterplot(data=df, x="Karyawan", y="Modal", hue="Jenis Usaha", s=80)
        plt.title("Hubungan Karyawan dan Modal Usaha", fontsize=12)
        plt.xlabel("Jumlah Karyawan", fontsize=10)
        plt.ylabel("Modal (Rp)", fontsize=10)
        st.pyplot(fig2)

    with tab3:
        st.markdown("### Korelasi antar Variabel")
        corr = df.select_dtypes(include="number").corr()
        fig3, ax3 = plt.subplots(figsize=(15, 5)) 
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Heatmap Korelasi Fitur", fontsize=12)
        st.pyplot(fig3)
