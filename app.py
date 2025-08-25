import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Membaca data dummy
data = pd.read_csv('dengue_dummy_data_with_puddle.csv')

# Menampilkan data awal untuk verifikasi
print(data.head())

# Mengonversi kategori 'Puddle' menjadi nilai numerik
data['Puddle'] = data['Puddle'].map({'None': 0, 'Low': 1, 'Medium': 2, 'High': 3})

# Memisahkan fitur (X) dan label (y)
X = data[['Body Temperature', 'Nausea Vomiting', 'Joint Pain', 'Appetite Loss', 'Dizziness', 'Red Spots', 'Puddle']]
y = data['Diagnosis']  # Diagnosis: 1 untuk DBD, 0 untuk bukan DBD

# Normalisasi data fitur
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Membagi data menjadi data pelatihan dan data pengujian (70% pelatihan, 30% pengujian)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Membangun model ANN
model = Sequential()

# Input layer dan hidden layer pertama
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))  # Dropout untuk menghindari overfitting

# Hidden layer kedua
model.add(Dense(32, activation='relu'))

# Output layer (sigmoid untuk klasifikasi biner)
model.add(Dense(1, activation='sigmoid'))

# Kompilasi model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Melatih model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy on Test Set: {accuracy*100:.2f}%")

# Menyimpan model setelah pelatihan
model.save('dengue_model_with_puddle.h5')

# Plotting grafik akurasi dan kerugian
plt.figure(figsize=(12, 6))

# Plot Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Kerugian
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
