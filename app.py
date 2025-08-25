import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Fungsi untuk memuat model
def load_trained_model():
    model_path = "/mnt/data/dengue_model_dummy.h5"
    model = load_model(model_path)
    return model

# Fungsi untuk melakukan prediksi
def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Tampilan UI untuk input data pengguna
st.title("Prediksi DBD dengan Model Dummy")

st.write("Masukkan data berikut untuk memprediksi apakah seseorang terdeteksi DBD:")

# Input data pengguna
body_temperature = st.number_input("Suhu Tubuh (°C)", min_value=36.0, max_value=42.0, value=37.5)
nausea_vomiting = st.number_input("Mual dan Muntah (0-5 kali)", min_value=0, max_value=5, value=1)
joint_pain = st.number_input("Nyeri Sendi (0-10 hari)", min_value=0, max_value=10, value=2)
appetite_loss = st.number_input("Kehilangan Nafsu Makan (0-10 hari)", min_value=0, max_value=10, value=3)
dizziness = st.number_input("Pusing (0-10 hari)", min_value=0, max_value=10, value=1)
red_spots = st.selectbox("Ruam (Red Spots)", options=['None', 'Little', 'Many'], index=0)
puddle = st.selectbox("Genangan Air (Puddle)", options=['None', 'Low', 'Medium', 'High'], index=0)

# Mengonversi input ke format yang bisa diterima oleh model
input_data = np.array([[body_temperature, nausea_vomiting, joint_pain, appetite_loss, dizziness, 
                        0 if red_spots == 'None' else 1 if red_spots == 'Little' else 2, 
                        0 if puddle == 'None' else 1 if puddle == 'Low' else 2 if puddle == 'Medium' else 3]])

# Normalisasi input data sesuai dengan data yang digunakan saat pelatihan
scaler = MinMaxScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Tombol untuk melakukan prediksi
if st.button("Prediksi DBD"):
    model = load_trained_model()  # Memuat model dummy
    prediction = make_prediction(model, input_data_scaled)  # Melakukan prediksi

    # Tampilkan hasil prediksi
    if prediction > 0.5:
        st.success("Prediksi: Positif DBD (Kemungkinan tinggi)")
    else:
        st.success("Prediksi: Negatif DBD (Kemungkinan rendah)")
