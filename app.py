import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="wide")

@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, scaler, class_names

def preprocess_image(image, target_size=(256, 256)):
    img_resized = cv2.resize(image, target_size)
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    lower_yellow = np.array([15, 30, 30])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    lower_dark = np.array([35, 20, 20])
    upper_dark = np.array([85, 255, 150])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    mask = cv2.bitwise_or(mask_green, mask_yellow)
    mask = cv2.bitwise_or(mask, mask_dark)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask_filtered = np.zeros_like(mask)
        cv2.drawContours(mask_filtered, [largest_contour], -1, 255, -1)
        mask = mask_filtered
    
    segmented = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    return img_resized, hsv, mask, segmented

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
        glcm = graycomatrix(gray_masked, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
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
    if len(contours) == 0:
        return np.array([0, 0, 0, 0, 0])
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
    area_normalized = area / img_area
    perimeter_normalized = perimeter / (2 * (mask.shape[0] + mask.shape[1]))
    return np.array([area_normalized, perimeter_normalized, aspect_ratio, circularity, solidity])

def extract_all_features(image):
    orig, hsv, mask, segmented = preprocess_image(image)
    color_features = extract_color_features(hsv, mask)
    texture_features = extract_glcm_features(segmented, mask)
    shape_features = extract_shape_features(mask)
    return np.concatenate([color_features, texture_features, shape_features])

def calculate_severity_score(image):
    orig, hsv, mask, segmented = preprocess_image(image)
    total_leaf_pixels = np.sum(mask > 0)
    if total_leaf_pixels == 0:
        return 0, "No Leaf", "gray", None
    lower_diseased = np.array([5, 30, 30])
    upper_diseased = np.array([25, 255, 255])
    diseased_mask = cv2.inRange(hsv, lower_diseased, upper_diseased)
    lower_necrotic = np.array([0, 0, 0])
    upper_necrotic = np.array([180, 255, 60])
    necrotic_mask = cv2.inRange(hsv, lower_necrotic, upper_necrotic)
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    total_diseased = cv2.bitwise_or(diseased_mask, necrotic_mask)
    total_diseased = cv2.bitwise_or(total_diseased, yellow_mask)
    total_diseased = cv2.bitwise_and(total_diseased, mask)
    kernel = np.ones((3, 3), np.uint8)
    total_diseased = cv2.morphologyEx(total_diseased, cv2.MORPH_OPEN, kernel)
    diseased_pixels = np.sum(total_diseased > 0)
    severity_score = (diseased_pixels / total_leaf_pixels) * 100
    if severity_score < 5:
        return severity_score, "Healthy/Minimal", "green", total_diseased
    elif severity_score < 15:
        return severity_score, "Mild", "yellow", total_diseased
    elif severity_score < 35:
        return severity_score, "Moderate", "orange", total_diseased
    else:
        return severity_score, "Severe", "red", total_diseased

def generate_explanation_heatmap(image, model, scaler, class_names, grid_size=6):
    orig, hsv, mask, segmented = preprocess_image(image)
    baseline_features = extract_all_features(image)
    baseline_scaled = scaler.transform(baseline_features.reshape(1, -1))
    baseline_pred_idx = model.predict(baseline_scaled)[0]
    if hasattr(model, "predict_proba"):
        baseline_proba = model.predict_proba(baseline_scaled)[0]
        baseline_confidence = baseline_proba[baseline_pred_idx]
    else:
        baseline_confidence = 1.0
    h, w = orig.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    importance_map = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            occluded = orig.copy()
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            occluded[y1:y2, x1:x2] = [128, 128, 128]
            occluded_features = extract_all_features(occluded)
            occluded_scaled = scaler.transform(occluded_features.reshape(1, -1))
            if hasattr(model, "predict_proba"):
                occluded_proba = model.predict_proba(occluded_scaled)[0]
                occluded_confidence = occluded_proba[baseline_pred_idx]
                importance_map[i, j] = max(0, baseline_confidence - occluded_confidence)
            else:
                occluded_pred = model.predict(occluded_scaled)[0]
                importance_map[i, j] = 1.0 if occluded_pred != baseline_pred_idx else 0.0
    if importance_map.max() > importance_map.min():
        importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min())
    importance_map_resized = cv2.resize(importance_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return importance_map_resized, class_names[baseline_pred_idx], baseline_confidence

def format_class_name(class_name):
    parts = class_name.split("___")
    if len(parts) == 2:
        crop, condition = parts
        crop = crop.replace("_", " ").replace("(", "").replace(")", "")
        condition = condition.replace("_", " ")
        return f"{crop} - {condition}"
    return class_name.replace("_", " ")

def main():
    st.title("🌿 Crop Leaf Disease Detection")
    st.markdown("Upload a leaf image to detect diseases in **Tomato**, **Potato**, or **Corn/Maize** plants.")
    
    try:
        model, scaler, class_names = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Analyze Leaf", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    features = extract_all_features(img_array)
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prediction_idx = model.predict(features_scaled)[0]
                    prediction = class_names[prediction_idx]
                    
                    confidence = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features_scaled)[0]
                        confidence = proba[prediction_idx]
                    
                    severity_score, severity_class, severity_color, diseased_mask = calculate_severity_score(img_array)
                    
                    st.markdown("---")
                    st.subheader("📊 Diagnosis Results")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        formatted_prediction = format_class_name(prediction)
                        if "healthy" in prediction.lower():
                            st.success(f"✅ {formatted_prediction}")
                        else:
                            st.warning(f"⚠️ {formatted_prediction}")
                    
                    with result_col2:
                        if confidence:
                            st.metric("Confidence", f"{confidence:.1%}")
                    
                    with result_col3:
                        if severity_color == "green":
                            st.success(f"🟢 {severity_class}")
                        elif severity_color == "yellow":
                            st.warning(f"🟡 {severity_class}")
                        elif severity_color == "orange":
                            st.warning(f"🟠 {severity_class}")
                        else:
                            st.error(f"🔴 {severity_class}")
                        st.caption(f"Severity: {severity_score:.1f}%")
                    
                    st.markdown("---")
                    st.subheader("📈 Severity Analysis")
                    
                    sev_col1, sev_col2 = st.columns(2)
                    
                    with sev_col1:
                        if diseased_mask is not None:
                            orig, _, _, _ = preprocess_image(img_array)
                            overlay = orig.copy()
                            overlay[diseased_mask > 0] = [0, 0, 255]
                            result_img = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)
                            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), 
                                    caption="Diseased Regions (red)", use_container_width=True)
                    
                    with sev_col2:
                        st.markdown(f"**Severity Score: {severity_score:.1f}%**")
                        st.progress(min(severity_score / 100, 1.0))
                    
                    st.markdown("---")
                    st.subheader("🔍 Explainable AI")
                    
                    with st.spinner("Generating heatmap..."):
                        importance_map, _, _ = generate_explanation_heatmap(img_array, model, scaler, class_names, grid_size=6)
                        
                        if importance_map is not None:
                            exp_col1, exp_col2 = st.columns(2)
                            
                            with exp_col1:
                                fig, ax = plt.subplots(figsize=(6, 6))
                                im = ax.imshow(importance_map, cmap="jet", vmin=0, vmax=1)
                                ax.set_title("Region Importance")
                                ax.axis("off")
                                plt.colorbar(im, ax=ax, fraction=0.046)
                                st.pyplot(fig)
                                plt.close()
                            
                            with exp_col2:
                                orig, _, _, _ = preprocess_image(img_array)
                                heatmap = cv2.applyColorMap((importance_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                                overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
                                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Explanation Overlay", use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("Traditional ML with handcrafted features")
        st.markdown("**Novel Features:**")
        st.markdown("- 📊 Severity Scoring")
        st.markdown("- 🔍 Explainable AI")

if __name__ == "__main__":
    main()
