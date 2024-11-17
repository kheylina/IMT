import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Judul Aplikasi
st.title('Indeks Masa Tubuh')

# Memuat data
try:
    df = pd.read_csv('bmi_train.csv')  # Pastikan file ini ada di direktori kerja
except FileNotFoundError:
    st.error("File 'bmi_train.csv' tidak ditemukan. Harap periksa lokasi file.")

# Mengganti Gender dengan angka (Male: 0, Female: 1)
df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})

# Split data untuk pelatihan model
X = df[['Height', 'Weight']]
y = df['Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Pilihan Opsi
option = st.sidebar.selectbox(
    'Opsi:',
    ('Indeks Masa Tubuh', 'DataScience')
)

# Opsi: Indeks Masa Tubuh
if option == 'Indeks Masa Tubuh':
    st.write("Silakan periksa Indeks Masa Tubuh Anda.")

    # Input untuk tinggi badan dan berat badan
    Height = st.number_input('Tinggi Badan (cm)', min_value=0.0, max_value=200.0, value=167.0)
    Weight = st.number_input('Berat Badan (kg)', min_value=0.0, max_value=200.0, value=49.0)   

    # Prediksi
    if st.button('Prediksi'):
        # Siapkan data untuk prediksi
        input_data = np.array([[Height, Weight]])
        prediction = knn.predict(input_data)

        # Mapping hasil prediksi
        index_levels = {0: 'Sangat Lemah', 1: 'Lemah', 2: 'Normal', 3: 'Kelebihan Berat', 4: 'Obesitas', 5: 'Obesitas Ekstrem'}
        index_level = index_levels[prediction[0]]
        st.write(f'Indeks Masa Tubuh Anda adalah: {index_level}')

# Opsi: DataScience
elif option == 'DataScience':
    st.write("Tentang Data")
    st.write("""
    Data ini diambil dari Kaggle, kita akan bekerja dengan sebuah dataset 
             yang berisi informasi tentang tinggi badan, 
             berat badan, jenis kelamin, dan indeks massa tubuh (IMT) individu.

IMT adalah ukuran yang menggunakan tinggi dan berat badan Anda untuk menentukan apakah berat badan Anda sehat.
Perhitungan IMT membagi berat badan seseorang (dalam kilogram) dengan tinggi badan mereka (dalam meter kuadrat).

Dataset ini berisi kolom-kolom berikut:
- **Gender (Jenis Kelamin):** Jenis kelamin individu.
- **Height (Tinggi Badan):** Tinggi badan individu dalam sentimeter.
- **Weight (Berat Badan):** Berat badan individu dalam kilogram.
- **Index (Indeks):** Indeks IMT individu, yang dikategorikan sebagai berikut:
    - 0: Sangat Lemah
    - 1: Lemah
    - 2: Normal
    - 3: Kelebihan Berat Badan
    - 4: Obesitas
    - 5: Obesitas Ekstrem
    """)

    st.divider()
    st.write("DataFrame")
    st.dataframe(df.head())
    st.write("Data Summary")
    st.dataframe(df.describe())
    st.write(""" 
- Tinggi badan Paling rendah adalah 140 cm,
- Berat badan paling rendah adalah 50 kg,
- Tinggi badan Paling Tinggi adalah 199 cm,
- Berat badan paling Tinggi adalah 160 kg,
- Rata-rata Tinggi badan : 171cm,
- Rata-rata berat badan : 106kg         
""")
    
    # Visualisasi Data
    st.write("Data Visualization")
  
    # Menyiapkan data untuk visualisasi
    gender_index_count = df.groupby(['Index', 'Gender']).size().unstack(fill_value=0)

    # Membuat plot menggunakan matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))

    # Membuat stacked bar chart
    gender_index_count.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_xlabel('Index')  # Mengatur label sumbu x
    ax.set_ylabel('Jumlah')  # Mengatur label sumbu y
    ax.set_title('Perhitungan dan Persentase Gender berdasarkan Indeks')  # Mengatur judul grafik

    # Menambahkan label persentase
    total_counts = gender_index_count.sum(axis=1)  # Total untuk setiap bar (indeks)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()

        # Menghindari pembagian dengan nol
        if height > 0 and total_counts[int(x)] > 0:
            percentage = height / total_counts[int(x)] * 100
            ax.text(
                x + width / 2, y + height / 2, f'{percentage:.1f}%', 
                ha='center', va='center', color='white', fontsize=9
            )

    plt.tight_layout()  # Menyesuaikan tata letak agar lebih rapi
    st.pyplot(fig)  # Menampilkan grafik
    st.write("""
    - Untuk kategori lemah paling banyak wanita (58.3%)
    - Untuk kategori Normal paling banyak  wanita (91.7%) 
    - Untuk kategori Kelebihan berat paling banyak wanita (172.2%) 
    - Untuk kategori Obesitas paling banyak wanita (62.7%)
    - Untuk kategori Obesitas Ekstrem paling banyak wanita dan pria (83.3%)        
    """)

    # Distribusi Gender
    st.write("Distribusi Gender")
    gender_counts = df['Gender'].value_counts()
    fig2, ax = plt.subplots()
    gender_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], ax=ax)
    ax.set_title('Distribusi Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Jumlah')

    st.pyplot(fig2)

# Pesan Default
else:
    st.write("Pilih opsi yang tersedia.")
