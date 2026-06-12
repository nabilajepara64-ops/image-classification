# ============================================================
# KLASIFIKASI KUCING VS KELINCI
# MENGGUNAKAN TRANSFER LEARNING MobileNetV2
# ============================================================

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# 1. CEK & EXTRACT ZIP
# ============================================================
zip_path = "Pet.zip"  # Path lokal di folder yang sama dengan app.py

if not os.path.exists(zip_path):
    print(f"Error: File '{zip_path}' tidak ditemukan. Pastikan file zip berada di folder yang sama.")
    exit()

print("Mengekstrak dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./dataset")
print("Dataset berhasil diextract")

# ============================================================
# 2. RESIZE SEMUA GAMBAR
# ============================================================
input_folder = "./dataset"
output_folder = "./dataset_resize"
classes = ["Kelinci", "Kucing"]
IMG_SIZE = (224, 224)

for kelas in classes:
    os.makedirs(os.path.join(output_folder, kelas), exist_ok=True)
    folder_kelas = os.path.join(input_folder, kelas)
    
    if not os.path.exists(folder_kelas):
        print(f"Warning: Folder kelas '{folder_kelas}' tidak ditemukan.")
        continue

    for filename in os.listdir(folder_kelas):
        img_path = os.path.join(folder_kelas, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            # Resize lebih optimal
            img = img.resize(IMG_SIZE)
            save_path = os.path.join(output_folder, kelas, filename)
            img.save(save_path)
        except Exception as e:
            print(f"Gagal membaca/menyimpan: {img_path}. Error: {e}")

print("Resize selesai")

# ============================================================
# 3. DATA AUGMENTATION
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# ============================================================
# 4. LOAD DATASET
# ============================================================
train_data = train_datagen.flow_from_directory(
    output_folder,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    output_folder,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("\nLabel Dataset:")
print(train_data.class_indices)

# ============================================================
# 5. LOAD BASE MODEL MobileNetV2 & MEMBUAT MODEL
# ============================================================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze layer pretrained
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()

# ============================================================
# 6. COMPILE & TRAINING MODEL
# ============================================================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

import streamlit as st

print("\nMemulai proses Training...")

# Bungkus proses training dengan spinner Streamlit
with st.spinner("Model sedang di-training... Mohon tunggu sampai 20 Epoch selesai."):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20
    )

st.success("Training Selesai!")

# ============================================================
# 7. SIMPAN MODEL
# ============================================================
model_save_path = "model_pet_cnn.keras"
model.save(model_save_path)
print(f"\nModel berhasil disimpan ke '{model_save_path}'")

# ============================================================
# 8. INTERFACE STREAMLIT UNTUK PREDIKSI
# ============================================================
st.write("---") # Membuat garis pembatas di web
st.header("🔮 Uji Prediksi Gambar")

# Membuat tombol upload gambar di halaman web Streamlit
uploaded_file = st.file_uploader("Pilih gambar kucing atau kelinci...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka dan menampilkan gambar yang di-upload ke layar web
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diunggah", use_container_width=True)
    
    # Preprocess gambar agar sesuai dengan input MobileNetV2
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Melakukan prediksi dengan model
    with st.spinner("Sedang menganalisis gambar..."):
        hasil = model.predict(img_array)
        nilai = hasil[0][0]
    
    # Menampilkan Hasil Prediksi ke layar web
    st.subheader("Hasil Analisis:")
    if nilai >= 0.5:
        st.error(f"Prediksi: **KUCING** (Confidence: {round(nilai * 100, 2)}%)")
    else:
        st.success(f"Prediksi: **KELINCI** (Confidence: {round((1 - nilai) * 100, 2)}%)")

# Menjaga agar window plot matplotlib tidak langsung tertutup di akhir program
plt.show()
