import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =========================
# Memuat Data dari CSV
# =========================
@st.cache
def load_data(csv_file: str):
    df = pd.read_csv(csv_file)
    return df

# =========================
# Aplikasi Streamlit
# =========================
st.title("Aplikasi Prediksi DBD dengan Random Forest")
st.sidebar.header("Masukkan Data Pasien")

# Memuat Data Dummy dari CSV
data_file = "data/data_dummy.csv"  # Ganti dengan path file CSV yang sesuai
df = load_data(data_file)

# Normalisasi data (kecuali kolom target 'Diagnosis')
scaler = StandardScaler()
df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']] = scaler.fit_transform(
    df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']]
)

# Mengubah kolom 'Red Spots' menjadi nilai numerik
df['Red Spots'] = df['Red Spots'].map({'None': 0, 'Little': 1, 'Many': 2})

# Membagi data menjadi fitur (X) dan target (y)
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Melatih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Input Data untuk Streamlit
temp = st.sidebar.number_input("Suhu Tubuh", 35.0, 42.0)
nausea = st.sidebar.number_input("Mual (Episode)", 0, 10)
joint_pain = st.sidebar.number_input("Nyeri Sendi (Hari)", 0, 7)
lack_of_appetite = st.sidebar.number_input("Kurang Nafsu Makan (Hari)", 0, 7)
dizziness = st.sidebar.number_input("Pusing (Hari)", 0, 7)
red_spots = st.sidebar.selectbox("Ruam Merah", ['None', 'Little', 'Many'])
water_stagnation = st.sidebar.radio("Genangan Air di Sekitar Rumah", ['Tidak', 'Ya'])

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

# Menampilkan hasil prediksi
if prediction == 1:
    st.write("Prediksi: **DBD Terdeteksi**")
else:
    st.write("Prediksi: **Tidak DBD**")
