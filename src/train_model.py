import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# DATA PATH
# Pastikan DATA_PATH mengarah ke lokasi file yang benar dari lokasi skrip ini
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "modal_usaha.csv"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "model", "model.pkl"
)


def train_model():
    # Load Dataset
    try:
        df = pd.read_csv(DATA_PATH)
        print("Dataset Loaded successfully!")
        print(df.head())
    except FileNotFoundError:
        print(f"ERROR: File dataset tidak ditemukan di {DATA_PATH}")
        return

    # encode kolom kategori
    le_jenis = LabelEncoder()
    le_lokasi = LabelEncoder()
    # Pastikan nama kolom sudah benar (case-sensitive)
    df["Jenis Usaha"] = le_jenis.fit_transform(df["Jenis Usaha"])
    df["Lokasi"] = le_lokasi.fit_transform(df["Lokasi"])

    # Memisahkan fitur dan target
    # Pastikan 'Modal' adalah nama kolom yang benar untuk target Anda
    x = df.drop("Modal", axis=1)
    y = df["Modal"]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Membuat dan Melatih Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # 🌟 PERBAIKAN FATAL: Menggunakan y_train sebagai target pelatihan
    model.fit(x_train, y_train)

    # Evaluasi Model
    y_pred = model.predict(x_test)

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)

    # Koefisien determinasi
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 30)
    print("Model Training Complete!")
    print(f"MAE: {mae:.2f}")  # Format diperbaiki
    print(f"R2 Score: {r2:.2f}")
    print("=" * 30)

    # Simpan model dan encoder
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {"model": model, "le_jenis": le_jenis, "le_lokasi": le_lokasi}, MODEL_PATH
    )


# Menjalankan fungsi train_model
if __name__ == "__main__":
    train_model()
