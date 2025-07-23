# app_plant.py
import streamlit as st
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import local_binary_pattern, hog

# Load models
species_model = joblib.load("models/species_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Feature extraction
def extract_features(image):
    img = cv2.resize(image, (128, 128))
    features = []

    avg_color = np.mean(img, axis=(0, 1))
    features.extend(avg_color)

    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
    features.extend(lbp_hist)

    gray_resized = cv2.resize(gray, (64, 128))
    hog_features = hog(gray_resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    features.extend(hog_features)

    return np.array(features)

# UI setup
st.set_page_config(page_title="🌿 Leaf Classifier", layout="centered")
st.title("🌿 Smart Leaf Classifier")
st.caption("🔍 SVM + handcrafted features")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    with st.spinner("Analyzing..."):
        features = extract_features(image)
        scaled = scaler.transform([features])
        pred_class = species_model.predict(scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        proba = species_model.predict_proba(scaled)[0]
        class_names = label_encoder.inverse_transform(np.arange(len(proba)))

        st.success(f"🧠 **Predicted Class:** `{pred_label}`")

        st.markdown("### 📊 Confidence Levels")
        fig, ax = plt.subplots()
        ax.barh(class_names, proba, color="teal")
        ax.set_xlabel("Probability")
        ax.invert_yaxis()
        st.pyplot(fig)
