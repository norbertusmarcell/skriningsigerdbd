import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Memuat data dummy dari file yang telah diunduh
df = pd.read_csv('data/data_dummy.csv')

# Pastikan kolom yang digunakan untuk normalisasi hanya berisi data numerik
df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']] = df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']].apply(pd.to_numeric, errors='coerce')

# Mengisi nilai NaN dengan rata-rata kolom yang sesuai
df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']] = df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']].fillna(df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']].mean())

# Melakukan normalisasi
scaler = StandardScaler()
df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']] = scaler.fit_transform(df[['Body Temperature', 'Nausea', 'Joint Pain', 'Lack of Appetite', 'Dizziness']])

# Memeriksa hasil data yang telah dinormalisasi
df.head()
