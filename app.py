import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Fungsi untuk membuat model dummy
def create_dummy_model():
    # Membangun model ANN
    model = Sequential([
        Dense(64, activation='relu', input_dim=6),  # Sesuaikan input dimensi dengan data dummy
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Kompilasi model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Simulasi pelatihan (gunakan data dummy atau sesuaikan dengan data yang ada)
    # Misalnya kita buat data dummy
    X = np.random.rand(100, 6)  # 100 sampel, 6 fitur
    y = np.random.randint(0, 2, size=(100, 1))  # 100 label biner (0 atau 1)

    # Latih model dengan data dummy
    model.fit(X, y, epochs=10, batch_size=16)

    # Menyimpan model dalam format .h5
    model_path = "/mnt/data/dengue_model_dummy.h5"
    model.save(model_path)
    return model_path

# Tampilan UI
st.title("Aplikasi Model Dummy DBD")

# Tombol untuk membuat model dummy
if st.button('Buat Model Dummy'):
    model_path = create_dummy_model()
    st.success(f'Model dummy telah dibuat dan disimpan di: {model_path}')

    # Menyediakan tombol untuk mendownload model
    with open(model_path, 'rb') as model_file:
        st.download_button(
            label="Download Model Dummy",
            data=model_file,
            file_name="dengue_model_dummy.h5",
            mime="application/octet-stream"
        )

# Opsional: tampilkan info model atau tombol lainnya
st.info("Setelah menekan tombol 'Buat Model Dummy', Anda dapat mendownload file model yang telah disimpan.")
