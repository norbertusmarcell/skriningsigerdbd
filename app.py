import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Menampilkan judul aplikasi
st.title("Pelatihan Model Deteksi DBD")

# Mengunggah file dataset
uploaded_file = st.file_uploader("Pilih file CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Membaca data dari file CSV
    data = pd.read_csv(uploaded_file)

    # Menampilkan beberapa data awal untuk melihat struktur
    st.write(data.head())

    # Memilih fitur dan label (sesuaikan dengan kolom dataset Anda)
    X = data[['Body Temperature', 'Nausea Vomiting', 'Joint Pain', 'Appetite Loss', 'Dizziness', 'Red Spots']]
    y = data['Diagnosis']  # 1 untuk positif DBD, 0 untuk negatif

    # Normalisasi data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Membagi data menjadi pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Menambahkan tombol untuk memulai pelatihan model
    if st.button("Mulai Pelatihan Model"):
        with st.spinner('Melatih model...'):
            # Membangun model ANN
            model = Sequential()
            model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
            model.add(Dropout(0.5))  # Dropout untuk mencegah overfitting
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))  # Output untuk klasifikasi biner

            # Kompilasi model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Melatih model
            history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

            # Evaluasi model
            loss, accuracy = model.evaluate(X_test, y_test)
            st.success(f"Model selesai dilatih dengan akurasi: {accuracy*100:.2f}%")

            # Menyimpan model
            model.save('dengue_model.h5')
            st.write("Model disimpan sebagai 'dengue_model.h5'.")

            # Plot akurasi pelatihan
            st.subheader('Grafik Akurasi')
            st.line_chart(history.history['accuracy'])

            # Plot kerugian pelatihan
            st.subheader('Grafik Kerugian')
            st.line_chart(history.history['loss'])
