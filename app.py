import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Memuat model dummy yang sudah disimpan
model = load_model("model/dengue_model.h5")

# Data dummy untuk pengujian (gunakan normalisasi yang sama seperti pada data asli)
scaler = MinMaxScaler()

# Judul aplikasi
st.title("Skrining Demam Berdarah (DBD) dengan Model Dummy")

# Formulir untuk input data gejala pasien
with st.form("prediction_form"):
    st.subheader("Masukkan Gejala Pasien")

    # Input gejala pasien
    temperature = st.slider("Suhu Tubuh (°C)", 36, 42, 37)
    nausea = st.slider("Mual (kali)", 0, 5, 1)
    joint_pain = st.slider("Nyeri Sendi (hari)", 0, 7, 2)
    appetite_loss = st.slider("Kehilangan Nafsu Makan (hari)", 0, 7, 3)
    dizziness = st.slider("Pusing (hari)", 0, 7, 2)
    gender = st.radio("Jenis Kelamin", ['L', 'P'])
    red_spots = st.radio("Bercak Merah (ruam)", ['Slight', 'Medium', 'Many'])
    puddles = st.radio("Apakah ada genangan air di sekitar rumah?", ['No', 'Yes'])

    # Tombol submit untuk prediksi
    submit_button = st.form_submit_button("Prediksi")

# Preprocessing data input setelah tombol submit
if submit_button:
    # Encoding variabel kategori menjadi numerik
    input_data = np.array([[temperature, nausea, joint_pain, appetite_loss, dizziness,
                            0 if gender == 'L' else 1, {'Slight': 0, 'Medium': 1, 'Many': 2}[red_spots], 
                            1 if puddles == 'Yes' else 0]])

    # Normalisasi data
    input_data_normalized = scaler.fit_transform(input_data)

    # Melakukan prediksi dengan model dummy
    prediction = model.predict(input_data_normalized)

    # Menampilkan hasil prediksi
    if prediction > 0.5:
        st.write("Prediksi: **Positif Demam Berdarah**")
    else:
        st.write("Prediksi: **Negatif Demam Berdarah**")

