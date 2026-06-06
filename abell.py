import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. EKSTRAKSI DATASET
local_zip = 'Pet.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/dataset_pet')
zip_ref.close()

# 2. DEFINISI DIREKTORI
base_dir = '/content/dataset_pet/Pet'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 3. DATA AUGMENTASI & PREPROCESSING
# Menggunakan fungsi preprocess_input bawaan MobileNetV2
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Target size disesuaikan dengan arsitektur standar MobileNetV2 (224x224)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Cek indeks kelas (0: Kucing, 1: Kelinci)
print("Indeks Kelas:", train_generator.class_indices)

# 4. MEMBANGUN MODEL TRANSFER LEARNING (MobileNetV2)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Membekukan bobot base model agar tidak ikut terlatih kembali
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Klasifikasi biner Kucing vs Kelinci
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. PELATIHAN MODEL (20 Epoch)
EPOCHS = 20
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // BATCH_SIZE)
)

# 6. MENYIMPAN MODEL KE FORMAT .KERAS
model.save('model_klasifikasi.keras')
print("Model berhasil disimpan dengan nama 'model_klasifikasi.keras'!")

# 7. VISUALISASI HASIL TRAINING
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# 8. FUNGSI PREDIKSI GAMBAR UJI
def prediksi_gambar(img_path):
    img = Image.open(img_path).resize(IMAGE_SIZE)
    img_array = np.array(img)
    
    # Memastikan format gambar RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    
    prediction = model.predict(img_preprocessed)[0][0]
    
    plt.imshow(img)
    plt.axis('off')
    
    if prediction < 0.5:
        plt.title(f"Prediksi: KUCING ({ (1 - prediction) * 100:.2f}%)")
        print(f"{img_path} adalah KUCING")
    else:
        plt.title(f"Prediksi: KELINCI ({prediction * 100:.2f}%)")
        print(f"{img_path} adalah KELINCI")
    plt.show()

# Eksekusi Uji Coba Gambar jika file tersedia
if os.path.exists('Gambar Uji1.jpg'):
    prediksi_gambar('Gambar Uji1.jpg')
if os.path.exists('Gambar Uji2.jpg'):
    prediksi_gambar('Gambar Uji2.jpg')
