# src/predict.py
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")


# Fungsi untuk memuat model
def load_model():
    model_data = joblib.load(MODEL_PATH)
    return model_data["model"], model_data["le_jenis"], model_data["le_lokasi"]


# Fungsi untuk melakukan prediksi
def predict_modal(jenis_usaha, lokasi, karyawan, harga_bahan, target_produksi, omset):
    model, le_jenis, le_lokasi = load_model()

    # Encode input kategorikal
    jenis_encoded = le_jenis.transform([jenis_usaha])[0]
    lokasi_encoded = le_lokasi.transform([lokasi])[0]

    # Buat DataFrame input
    data = pd.DataFrame(
        [
            {
                "Jenis Usaha": jenis_encoded,
                "Lokasi": lokasi_encoded,
                "Karyawan": karyawan,
                "Harga Bahan": harga_bahan,
                "Target Produksi": target_produksi,
                "Omset": omset,
            }
        ]
    )

    # Prediksi
    prediction = model.predict(data)[0]
    return round(prediction, 2)
