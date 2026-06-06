import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Klasifikasi Kucing vs Kelinci", layout="centered")

st.title("🐱 Klasifikasi Gambar Kucing vs Kelinci 🐰")
st.write("Unggah gambar untuk memprediksi apakah gambar tersebut merupakan Kucing atau Kelinci.")

# 1. LOAD MODEL YANG SUDAH JADI
# Fungsi ini akan menyimpan model di cache agar tidak perlu loading terus-menerus setiap halaman di-refresh
@st.cache_resource
def load_my_model():
    # Menggunakan compile=False karena model hanya digunakan untuk prediksi (Inference)
    return tf.keras.models.load_model('model_klasifikasi.keras', compile=False)

try:
    model = load_my_model()
    st.success("Model AI berhasil dimuat!")
except Exception as e:
    st.error("Gagal memuat model. Pastikan file 'model_klasifikasi.keras' sudah Anda upload ke repositori GitHub yang sama dengan file ini.")
    st.stop()

# 2. FITUR UNGGAH GAMBAR OLEH USER
uploaded_file = st.file_uploader("Pilih file gambar (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang dipilih user
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    st.write("🔄 Sedang menganalisis gambar...")
    
    # Preprocessing gambar sesuai standar MobileNetV2 (224x224)
    IMAGE_SIZE = (224, 224)
    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized)
    
    # Menghapus channel Alpha jika format gambar PNG memiliki 4 channel (RGBA)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    # Mengatasi jika gambar hitam putih / grayscale (hanya 2 dimensi)
    elif len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
        
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    
    # Melakukan Prediksi
    prediction = model.predict(img_preprocessed)[0][0]
    
    # Menampilkan Hasil Prediksi Berdasarkan Threshold 0.5
    st.markdown("---")
    if prediction < 0.5:
        persentase = (1 - prediction) * 100
        st.subheader(f"Hasil Analisis: KUCING 🐱")
        st.progress(int(persentase))
        st.write(f"Tingkat Keyakinan Model: **{persentase:.2f}%**")
    else:
        persentase = prediction * 100
        st.subheader(f"Hasil Analisis: KELINCI 🐰")
        st.progress(int(persentase))
        st.write(f"Tingkat Keyakinan Model: **{persentase:.2f}%**")
