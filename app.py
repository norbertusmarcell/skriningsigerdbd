import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model

# Judul aplikasi
st.title("Membuat Model Dummy untuk Prediksi Demam Berdarah (DBD)")

# Fungsi untuk membuat dan menyimpan model dummy
def create_dummy_model():
    # Membuat model dummy
    model = Sequential([
        Dense(64, input_dim=7, activation='relu'),  # 7 input features (seperti suhu tubuh, nyeri sendi, dll)
        Dense(32, activation='relu'),  # Lapisan tersembunyi
        Dense(1, activation='sigmoid')  # Output layer (0 atau 1)
    ])

    # Inisialisasi bobot secara acak
    model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

    # Menyimpan model dummy ke dalam file .h5
    model_path = "/mnt/data/dengue_model_dummy.h5"
    model.save(model_path)
    return model_path

# Tombol untuk membuat model
if st.button('Buat Model Dummy'):
    model_path = create_dummy_model()
    st.success(f'Model dummy telah dibuat dan disimpan! [Download Model](/{model_path})')

    # Menyediakan link download model
    st.download_button(
        label="Download Model .h5",
        data=model_path,
        file_name="dengue_model_dummy.h5",
        mime="application/octet-stream"
    )
