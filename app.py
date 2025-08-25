import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model

# Title for the Streamlit app
st.title("Create and Download Dummy Model for DBD Prediction")

# Function to create and save the model
def create_and_save_model():
    # Create a dummy model
    model = Sequential([
        Dense(64, input_dim=7, activation='relu'),  # 7 input features (temperature, joint pain, etc.)
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification (positive/negative)
    ])

    # Initialize weights randomly
    model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

    # Save the model to an .h5 file
    model_path = "/mnt/data/dengue_model_dummy.h5"
    model.save(model_path)
    
    return model_path

# Button to create the model
if st.button("Create Dummy Model"):
    model_path = create_and_save_model()
    st.success(f"Dummy model created and saved successfully! [Download Model](/{model_path})")

    # Provide download button for the model
    st.download_button(
        label="Download Model .h5",
        data=model_path,
        file_name="dengue_model_dummy.h5",
        mime="application/octet-stream"
    )
