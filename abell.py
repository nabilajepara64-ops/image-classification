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

print("\nMemulai proses Training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# ============================================================
# 7. SIMPAN MODEL
# ============================================================
model_save_path = "model_pet_cnn.keras"
model.save(model_save_path)
print(f"\nModel berhasil disimpan ke '{model_save_path}'")

# ============================================================
# 8. FUNGSI PREDIKSI
# ============================================================
def prediksi_gambar(path_gambar):
    if not os.path.exists(path_gambar):
        print(f"File gambar uji '{path_gambar}' tidak ditemukan.")
        return

    img = load_img(path_gambar, target_size=(224, 224))
    
    # Menampilkan gambar secara lokal (akan membuka jendela baru)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediksi untuk: {os.path.basename(path_gambar)}")
    plt.show(block=False)  # Gunakan block=False agar tidak menghentikan jalannya kode

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess khusus MobileNetV2
    img_array = preprocess_input(img_array)
    hasil = model.predict(img_array)
    nilai = hasil[0][0]

    print("Nilai Prediksi :", nilai)

    # 0 = Kelinci, 1 = Kucing
    if nilai >= 0.5:
        print("Prediksi : KUCING")
        print("Confidence :", round(nilai * 100, 2), "%")
    else:
        print("Prediksi : KELINCI")
        print("Confidence :", round((1 - nilai) * 100, 2), "%")

# ============================================================
# 9. PREDIKSI GAMBAR UJI
# ============================================================
print("\n==========================")
print("HASIL PREDIKSI")
print("==========================")

print("\nGambar Uji 1")
prediksi_gambar("Gambar Uji1.jpg")

print("\nGambar Uji 2")
prediksi_gambar("Gambar Uji2.jpg")

# Menjaga agar window plot matplotlib tidak langsung tertutup di akhir program
plt.show()
