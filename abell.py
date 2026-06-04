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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# 1. EXTRACT ZIP
# ============================================================
# Pastikan file 'Pet.zip' berada di folder yang sama dengan app.py
zip_path = "Pet.zip"
dataset_dir = "dataset"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Dataset berhasil diextract.")
else:
    print(f"Peringatan: File {zip_path} tidak ditemukan. Pastikan dataset sudah siap di folder '{dataset_dir}'.")

# ============================================================
# 2. RESIZE SEMUA GAMBAR
# ============================================================
input_folder = dataset_dir
output_folder = "dataset_resize"
classes = ["Kelinci", "Kucing"]
IMG_SIZE = (224, 224)

if os.path.exists(input_folder):
    for kelas in classes:
        os.makedirs(os.path.join(output_folder, kelas), exist_ok=True)
        folder_kelas = os.path.join(input_folder, kelas)
        
        if not os.path.exists(folder_kelas):
            continue
            
        for filename in os.listdir(folder_kelas):
            img_path = os.path.join(folder_kelas, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                save_path = os.path.join(output_folder, kelas, filename)
                img.save(save_path)
            except:
                print("Gagal membaca:", img_path)
    print("Resize selesai.")
else:
    print(f"Error: Folder input '{input_folder}' tidak ditemukan untuk proses resize.")

# ============================================================
# 3. DATA AUGMENTATION & LOAD DATASET
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

if os.path.exists(output_folder):
    train_data = train_datagen.flow_from_directory(
        output_folder,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_data = train_datagen.flow_from_directory(
        output_folder,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    print("\nLabel Dataset:")
    print(train_data.class_indices)
    
    # ============================================================
    # 4. MEMBUAT & COMPILE MODEL
    # ============================================================
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.summary()

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # ============================================================
    # 5. TRAINING & SIMPAN MODEL
    # ============================================================
    print("\nMemulai proses training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20
    )

    model.save("model_pet_cnn.keras")
    print("\nModel berhasil disimpan dengan nama 'model_pet_cnn.keras'")
else:
    print(f"Error: Folder '{output_folder}' tidak ditemukan. Training dibatalkan.")

# ============================================================
# 6. FUNGSI PREDIKSI
# ============================================================
def prediksi_gambar(path_gambar):
    if not os.path.exists(path_gambar):
        print(f"File gambar tidak ditemukan: {path_gambar}")
        return

    img = load_img(path_gambar, target_size=IMG_SIZE)
    
    # Tampilkan gambar menggunakan matplotlib (akan memunculkan window baru di lokal)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    hasil = model.predict(img_array)
    nilai = hasil[0][0]

    print("Nilai Prediksi :", nilai)

    if nilai >= 0.5:
        print("\nPrediksi : KUCING")
        print("Confidence :", round(nilai * 100, 2), "%")
    else:
        print("\nPrediksi : KELINCI")
        print("Confidence :", round((1 - nilai) * 100, 2), "%")

# ============================================================
# 7. PREDIKSI GAMBAR UJI
# ============================================================
if 'model' in locals():
    print("\n==========================")
    print("HASIL PREDIKSI")
    print("==========================")

    # Pastikan file gambar uji ini ada di direktori yang sama
    print("\nGambar Uji 1")
    prediksi_gambar("Gambar Uji1.jpg")

    print("\nGambar Uji 2")
    prediksi_gambar("Gambar Uji2.jpg")