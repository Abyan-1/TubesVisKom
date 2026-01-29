import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Masker Wajah", page_icon="😷", layout="centered")

# --- JUDUL ---
st.title("😷 Deteksi Masker Wajah")
st.write("Arahkan wajah ke kamera. Pastikan cahaya cukup terang.")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_masker_fix.h5')
    return model

with st.spinner('Sedang memuat model AI...'):
    model = load_model()

# --- 2. MODEL DETEKSI WAJAH ---
# Menggunakan versi alt2 agar lebih sensitif pada wajah bermasker
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
except:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 3. INPUT KAMERA ---
img_file_buffer = st.camera_input("Ambil Foto Wajah")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    img_display = img_array.copy()
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))
    
    st.write(f"Terdeteksi **{len(faces)}** wajah.")

    if len(faces) == 0:
        st.warning("⚠️ Wajah tidak terdeteksi. Coba posisikan wajah di tengah dan cari cahaya terang.")

    for (x, y, w, h) in faces:
        # Preprocessing
        wajah_crop = img_array[y:y+h, x:x+w]
        wajah_resize = cv2.resize(wajah_crop, (150, 150))
        
        wajah_input = np.array(wajah_resize) / 255.0
        wajah_input = np.expand_dims(wajah_input, axis=0)

        # Prediksi
        prediction = model.predict(wajah_input)
        hasil_prediksi = prediction[0][0]

        # --- LOGIKA FINAL YANG BENAR ---
        # 0 = with_mask (Pakai Masker)
        # 1 = without_mask (Tidak Pakai)
        
        if hasil_prediksi < 0.5: 
            # Nilai mendekati 0 -> PAKAI MASKER
            label = "PAKAI MASKER"
            color = (0, 255, 0) # Hijau
            confidence = (1 - hasil_prediksi) * 100 # Hitung kepastian
        else:
            # Nilai mendekati 1 -> TIDAK PAKAI
            label = "TIDAK PAKAI MASKER"
            color = (255, 0, 0) # Merah
            confidence = hasil_prediksi * 100

        score_text = f"{confidence:.1f}%"

        # Gambar Kotak
        cv2.rectangle(img_display, (x, y), (x+w, y+h), color, 4)
        cv2.putText(img_display, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img_display, score_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    st.image(img_display, caption="Hasil Analisis AI", use_column_width=True)