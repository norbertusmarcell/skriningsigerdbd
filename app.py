import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# =========================
# Memuat Data dari CSV
# =========================
@st.cache_data
def load_data(csv_file: str):
    # Cek apakah file ada di path yang ditentukan
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df
    else:
        st.error(f"File tidak ditemukan di path: {csv_file}")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong jika file tidak ditemukan

# =========================
# Aplikasi Streamlit
# =========================
st.title("Aplikasi Prediksi DBD dengan Random Forest")
st.sidebar.header("Masukkan Data Pasien")

# Path file CSV
data_file = 'data/data_dummy.csv'  # Ganti dengan path yang sesuai jika berbeda

# Memuat Data Dummy dari CSV
df = load_data(data_file)

# Jika file berhasil dimuat, lanjutkan proses
if not df.empty:
    # Memastikan kolom 'Red Spots' menjadi numerik (konversi kategori ke nilai numerik)
    df['Red Spots'] = df['Red Spots'].map({'None': 0, 'Little': 1, 'Many': 2})

    # Menghapus baris yang mengandung nilai NaN
    df = df.dropna()

    # Normalisasi data (fitur numerik)
    scaler = StandardScaler()
    df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']] = df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']].apply(pd.to_numeric, errors='coerce')
    df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']] = scaler.fit_transform(df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']])

    # Membagi data menjadi fitur (X) dan target (y)
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Melatih model Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Membuat dua kolom untuk menampilkan form di samping
    col1, col2 = st.columns(2)

    with col1:
        temp = st.number_input("Suhu Tubuh", 35.0, 42.0)
        nausea = st.number_input("Mual (Episode)", 0, 10)
        joint_pain = st.number_input("Nyeri Sendi (Hari)", 0, 7)
        lack_of_appetite = st.number_input("Kurang Nafsu Makan (Hari)", 0, 7)
        dizziness = st.number_input("Pusing (Hari)", 0, 7)

    with col2:
        red_spots = st.selectbox("Ruam Merah", ['None', 'Little', 'Many'])
        water_stagnation = st.radio("Genangan Air di Sekitar Rumah", ['Tidak', 'Ya'])

    # Menyiapkan data input untuk prediksi
    input_data = pd.DataFrame({
        'Body Temperature': [temp],
        'Nausea': [nausea],
        'Joint Pain': [joint_pain],
        'Lack of Appetite': [lack_of_appetite],
        'Dizziness': [dizziness],
        'Red Spots': [red_spots],
        'Water Stagnation': [1 if water_stagnation == 'Ya' else 0]
    })

    # Normalisasi data input
    input_data[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']] = scaler.transform(input_data[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']])

    # Mengubah 'Red Spots' menjadi nilai numerik
    input_data['Red Spots'] = input_data['Red Spots'].map({'None': 0, 'Little': 1, 'Many': 2})

    # Melakukan prediksi
    prediction = model.predict(input_data)

    # Menampilkan hasil prediksi dengan warna
    if prediction == 1:
        st.markdown('<span style="color:red;">Prediksi: **DBD Terdeteksi**. Segera ke rumah sakit terdekat.</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:green;">Prediksi: **Tidak DBD**. Lakukan upaya pencegahan karena di kota Anda sedang tinggi kejadian DBDnya.</span>', unsafe_allow_html=True)

