import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ===== Konfigurasi halaman =====
st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Padi",
    page_icon="🌾",
    layout="centered"
)

# ===== Load model dan class mapping (cache supaya tidak reload tiap interaksi) =====
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_padi_final.keras')
    with open('class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    return model, class_mapping

model, class_mapping = load_model()

IMG_SIZE = 128

# ===== Deskripsi tiap kelas penyakit (untuk ditampilkan ke pengguna) =====
deskripsi_penyakit = {
    "Bacterialblight": "Penyakit hawar daun bakteri, ditandai garis kuning kecoklatan yang memanjang dari tepi daun.",
    "Blast": "Penyakit blast, ditandai bercak berbentuk belah ketupat dengan tepi coklat dan bagian tengah keabuan.",
    "Brownspot": "Penyakit bercak coklat, ditandai bercak bulat kecil berwarna coklat tersebar di permukaan daun.",
    "Tungro": "Penyakit tungro, ditandai daun menguning kecoklatan disertai pertumbuhan tanaman yang kerdil."
}

# ===== Header =====
st.title("🌾 Klasifikasi Penyakit Daun Padi")
st.markdown("""
Aplikasi ini menggunakan model **Convolutional Neural Network (CNN)** untuk mengklasifikasikan
penyakit pada daun padi berdasarkan citra. Model dapat mengenali 4 kategori:
**Bacterial Blight, Blast, Brown Spot, dan Tungro**.
""")

st.divider()

# ===== Upload gambar =====
uploaded_file = st.file_uploader(
    "Unggah citra daun padi (format JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # ===== Preprocessing =====
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ===== Prediksi =====
    with st.spinner("Menganalisis citra..."):
        prediction = model.predict(img_array)
        pred_idx = np.argmax(prediction[0])
        pred_class = class_mapping[str(pred_idx)]
        confidence = prediction[0][pred_idx] * 100

    with col2:
        st.subheader("Hasil Prediksi")
        st.success(f"**{pred_class}**")
        st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")

        st.markdown("**Penjelasan:**")
        st.info(deskripsi_penyakit.get(pred_class, "Deskripsi tidak tersedia."))

    # ===== Detail probabilitas semua kelas =====
    st.divider()
    st.subheader("Detail Probabilitas Tiap Kelas")

    for idx, prob in enumerate(prediction[0]):
        cls_name = class_mapping[str(idx)]
        st.write(f"{cls_name}")
        st.progress(float(prob))
        st.caption(f"{prob*100:.2f}%")

else:
    st.info("Silakan unggah gambar daun padi untuk memulai klasifikasi.")

st.divider()
st.caption("Model CNN dilatih menggunakan dataset Rice Leaf Disease Image (4.794 citra unik, 4 kelas) dengan akurasi pengujian 99.79%.")