import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# ============================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ============================================================
st.set_page_config(page_title="Klasifikasi Kucing vs Kelinci", layout="centered")

st.title("🐱 Klasifikasi Kucing vs Kelinci 🐰")
st.write("Unggah gambar hewan untuk memprediksi apakah itu Kucing atau Kelinci.")

# ============================================================
# 2. LOAD MODEL YANG SUDAH JADI (DENGAN CACHING)
# ============================================================
MODEL_PATH = "model_pet_cnn.keras"

@st.cache_resource
def load_my_model():
    # Fungsi ini memastikan model hanya di-load sekali ke memori
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

model = load_my_model()

# Jika file model belum di-upload ke GitHub
if model is None:
    st.error(f"File '{MODEL_PATH}' tidak ditemukan di repositori GitHub Anda. Silakan upload file modelnya terlebih dahulu!")
else:
    # ============================================================
    # 3. INTERFACES UPLOAD GAMBAR
    # ============================================================
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah ke web
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
        
        st.write("⏳ Sedang memproses prediksi...")

        # Pemrosesan gambar agar sesuai input model
        IMG_SIZE = (224, 224)
        img_resized = image.resize(IMG_SIZE)
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediksi menggunakan model
        hasil = model.predict(img_array)
        nilai = hasil[0][0]

        # ============================================================
        # 4. MENAMPILKAN HASIL PREDIKSI KE USER
        # ============================================================
        st.subheader("Hasil Analisis:")
        if nilai >= 0.5:
            confidence = round(nilai * 100, 2)
            st.success(f"Prediksi: **KUCING** ({confidence}%)")
        else:
            confidence = round((1 - nilai) * 100, 2)
            st.info(f"Prediksi: **KELINCI** ({confidence}%)")
