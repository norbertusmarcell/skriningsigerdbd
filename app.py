import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =========================
# Memuat Data dari CSV
# =========================
@st.cache_data
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

# Memastikan kolom 'Red Spots' menjadi numerik (konversi kategori ke nilai numerik)
df['Red Spots'] = df['Red Spots'].map({'None': 0, 'Little': 1, 'Many': 2})

# Menghapus baris yang mengandung nilai NaN
df = df.dropna()

# Menampilkan data yang telah diperbaiki
st.write("Data yang telah diperbaiki:", df.head())

# Normalisasi data (fitur numerik)
scaler = StandardScaler()
df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']] = df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']].apply(pd.to_numeric, errors='coerce')
df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']] = scaler.fit_transform(df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']])

# Membagi data menjadi fitur (X) dan target (y)
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Melatih model Random Forest
model = RandomForestClassifi

