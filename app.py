
import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Tomato & Potato Disease Doctor",
    page_icon="🍅",
    layout="wide"
)

# Load model artifacts
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, scaler, class_names

# ============== IMPROVED PREPROCESSING (WITH SHADOW REMOVAL) =============
def preprocess_image(image, target_size=(256, 256)):
    # Resize
    img_resized = cv2.resize(image, target_size)
    
    # --- SHADOW REMOVAL (CLAHE) ---
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_normalized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # --- SEGMENTATION ---
    img_blur = cv2.GaussianBlur(img_normalized, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([25, 255, 255])

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_green, upper_green),
                          cv2.inRange(hsv, lower_brown, upper_brown))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask_filtered = np.zeros_like(mask)
        cv2.drawContours(mask_filtered, [largest_contour], -1, 255, -1)
        mask = mask_filtered

    segmented = cv2.bitwise_and(img_normalized, img_normalized, mask=mask)

    return img_normalized, hsv, mask, segmented

# ============== FEATURE EXTRACTION =============
def extract_color_features(hsv_image, mask, bins=32):
    features = []
    for i in range(3):
        hist = cv2.calcHist([hsv_image], [i], mask, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)

def extract_glcm_features(image, mask):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    gray_masked = img_as_ubyte(gray_masked / 255.0) if gray_masked.max() > 1 else img_as_ubyte(gray_masked)

    try:
        glcm = graycomatrix(gray_masked, distances=[1, 2, 3],
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True, normed=True)

        contrast = graycoprops(glcm, "contrast").mean()
        correlation = graycoprops(glcm, "correlation").mean()
        energy = graycoprops(glcm, "energy").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        dissimilarity = graycoprops(glcm, "dissimilarity").mean()

        return np.array([contrast, correlation, energy, homogeneity, dissimilarity])
    except:
        return np.array([0, 0, 0, 0, 0])

def extract_shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return np.array([0, 0, 0, 0, 0])

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    img_area = mask.shape[0] * mask.shape[1]
    return np.array([area/img_area, perimeter/(2*(mask.shape[0]+mask.shape[1])), aspect_ratio, circularity, solidity])

def extract_all_features(image):
    orig, hsv, mask, segmented = preprocess_image(image)
    color_features = extract_color_features(hsv, mask)
    texture_features = extract_glcm_features(segmented, mask)
    shape_features = extract_shape_features(mask)
    return np.concatenate([color_features, texture_features, shape_features])

# ============== SEVERITY SCORING (Updated Logic) =============
def calculate_severity_score(image):
    orig, hsv, mask, segmented = preprocess_image(image)
    total_leaf_pixels = np.sum(mask > 0)
    if total_leaf_pixels == 0: return 0, "No Leaf", "gray", None
    
    # Blight specific detection (Dark brown/Black necrotic spots)
    lower_necrotic = np.array([0, 0, 0])
    upper_necrotic = np.array([180, 255, 80]) # Darker spots for blight
    
    # Yellow halos (Early Blight characteristic)
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    
    necrotic_mask = cv2.inRange(hsv, lower_necrotic, upper_necrotic)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    total_diseased = cv2.bitwise_or(necrotic_mask, yellow_mask)
    total_diseased = cv2.bitwise_and(total_diseased, mask)
    
    diseased_pixels = np.sum(total_diseased > 0)
    severity_score = (diseased_pixels / total_leaf_pixels) * 100
    
    if severity_score < 5: status, color = "Healthy/Low", "green"
    elif severity_score < 20: status, color = "Early Stage", "yellow"
    elif severity_score < 40: status, color = "Moderate", "orange"
    else: status, color = "Severe", "red"
    
    return severity_score, status, color, total_diseased

def format_class_name(class_name):
    # Cleaner formatting for the specific classes
    return class_name.replace("___", " - ").replace("_", " ")

# ============== MAIN APP =============
def main():
    st.title("🍅 Potato & Tomato Disease Doctor")
    st.markdown("Specialized detection for **Early Blight**, **Late Blight**, and **Healthy** crops.")

    try:
        model, scaler, class_names = load_model()
    except:
        st.error("Model files not found. Please train the model and save artifacts first.")
        return

    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        if len(img_array.shape) == 2: img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        if st.button("Diagnose Crop", type="primary"):
            with st.spinner("Analyzing textures and colors..."):
                # Prediction
                features = extract_all_features(img_array)
                features_scaled = scaler.transform(features.reshape(1, -1))
                pred_idx = model.predict(features_scaled)[0]
                prediction = class_names[pred_idx]
                
                # Probabilities
                probs = model.predict_proba(features_scaled)[0] if hasattr(model, "predict_proba") else None
                confidence = probs[pred_idx] if probs is not None else 1.0

                # Severity
                score, status, color, disease_mask = calculate_severity_score(img_array)

                # --- Results UI ---
                st.divider()
                r1, r2 = st.columns([2, 1])
                
                with r1:
                    fmt_pred = format_class_name(prediction)
                    if "healthy" in prediction.lower():
                        st.success(f"## ✅ {fmt_pred}")
                    else:
                        st.error(f"## ⚠️ {fmt_pred}")
                    st.info(f"Confidence: **{confidence:.1%}**")

                with r2:
                    st.metric("Severity Score", f"{score:.1f}%", status)
                
                # Visuals
                st.divider()
                v1, v2 = st.columns(2)
                with v1:
                    st.subheader("Disease Visualization")
                    orig, _, _, _ = preprocess_image(img_array)
                    overlay = orig.copy()
                    if disease_mask is not None:
                        overlay[disease_mask > 0] = [0, 0, 255] # Red highlight
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Detected Lesions")
                
                with v2:
                    if probs is not None:
                        st.subheader("Class Probabilities")
                        # Get top 3
                        top_idx = np.argsort(probs)[::-1][:3]
                        for i in top_idx:
                            st.write(f"**{format_class_name(class_names[i])}**: {probs[i]:.1%}")
                            st.progress(float(probs[i]))

if __name__ == "__main__":
    main()
