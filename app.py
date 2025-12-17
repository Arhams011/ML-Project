import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

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
        return 0, "No Leaf Detected", "gray", None
    
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    lower_diseased = np.array([5, 40, 50])
    upper_diseased = np.array([25, 255, 255])
    diseased_mask = cv2.inRange(hsv, lower_diseased, upper_diseased)
    
    lower_necrotic = np.array([5, 20, 0])
    upper_necrotic = np.array([25, 255, 80])
    necrotic_mask = cv2.inRange(hsv, lower_necrotic, upper_necrotic)
    
    lower_yellow = np.array([20, 50, 120])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    shadow_mask = np.zeros_like(mask)
    shadow_mask[(v < 60) & (s < 40)] = 255
    
    total_diseased = cv2.bitwise_or(diseased_mask, necrotic_mask)
    total_diseased = cv2.bitwise_or(total_diseased, yellow_mask)
    total_diseased = cv2.bitwise_and(total_diseased, cv2.bitwise_not(shadow_mask))
    total_diseased = cv2.bitwise_and(total_diseased, mask)
    
    kernel = np.ones((3, 3), np.uint8)
    total_diseased = cv2.morphologyEx(total_diseased, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(total_diseased, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        min_area = total_leaf_pixels * 0.01
        mask_cleaned = np.zeros_like(total_diseased)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask_cleaned, [contour], -1, 255, -1)
        total_diseased = mask_cleaned
    
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

def format_class_name(class_name):
    parts = class_name.split("___")
    if len(parts) == 2:
        crop, condition = parts
        crop = crop.replace("_", " ").replace("(", "").replace(")", "")
        condition = condition.replace("_", " ")
        return f"{crop} - {condition}"
    return class_name.replace("_", " ")

def get_treatment_recommendation(disease_name, severity_class):
    disease_lower = disease_name.lower()
    
    treatments = {
        "early_blight": {
            "description": "Fungal disease causing dark spots with concentric rings (target-like pattern)",
            "organic": ["Remove and destroy infected leaves immediately", "Apply copper-based fungicide (Bordeaux mixture)", "Spray neem oil solution (2-3 tbsp per gallon)", "Improve air circulation by pruning"],
            "chemical": ["Chlorothalonil (Daconil)", "Mancozeb", "Azoxystrobin (Quadris)"],
            "prevention": ["Rotate crops every 2-3 years", "Use drip irrigation instead of overhead watering", "Mulch around plants to prevent soil splash", "Plant resistant varieties"]
        },
        "late_blight": {
            "description": "Serious fungal disease causing water-soaked lesions that turn brown/black",
            "organic": ["Remove and BURN infected plants immediately", "Apply copper fungicide weekly during wet weather", "Spray with Bacillus subtilis (Serenade)", "Remove all nearby volunteer plants"],
            "chemical": ["Metalaxyl (Ridomil)", "Chlorothalonil", "Mancozeb + Metalaxyl combination"],
            "prevention": ["Plant certified disease-free seeds/tubers", "Avoid overhead irrigation", "Ensure good air circulation", "Destroy all infected plant material"]
        },
        "common_rust": {
            "description": "Fungal disease causing small reddish-brown pustules on leaves",
            "organic": ["Remove infected leaves early", "Apply sulfur-based fungicide", "Spray neem oil", "Improve plant spacing for airflow"],
            "chemical": ["Azoxystrobin", "Propiconazole", "Mancozeb"],
            "prevention": ["Plant resistant varieties", "Early planting to avoid peak rust season", "Proper plant spacing", "Remove crop debris after harvest"]
        },
        "northern_leaf_blight": {
            "description": "Fungal disease causing long, gray-green cigar-shaped lesions",
            "organic": ["Rotate crops for 2+ years", "Remove and destroy infected residue", "Apply copper-based fungicides"],
            "chemical": ["Pyraclostrobin", "Azoxystrobin", "Propiconazole"],
            "prevention": ["Use resistant hybrids", "Plow under crop residue", "Crop rotation with non-host crops", "Avoid excessive nitrogen fertilization"]
        },
        "healthy": {
            "description": "No disease detected - plant appears healthy!",
            "organic": ["Continue regular watering schedule", "Apply organic compost monthly", "Monitor for early signs of disease"],
            "chemical": ["No treatment needed"],
            "prevention": ["Maintain good air circulation", "Water at base of plant, not leaves", "Regular inspection of leaves", "Keep garden clean of debris"]
        }
    }
    
    matched_treatment = None
    
    for key in treatments:
        key_check = key.replace("_", "").lower()
        disease_check = disease_lower.replace("_", "").replace(" ", "")
        if key_check in disease_check:
            matched_treatment = treatments[key]
            break
    
    if severity_class == "Severe":
        urgency = "🔴 CRITICAL - Immediate Action Required"
    elif severity_class == "Moderate":
        urgency = "🟠 WARNING - Treatment Needed Soon"
    elif severity_class == "Mild":
        urgency = "🟡 CAUTION - Monitor and Treat Early"
    else:
        urgency = "🟢 HEALTHY - Preventive Care Only"
    
    if matched_treatment:
        return matched_treatment, urgency
    else:
        return {
            "description": f"Disease detected: {disease_name.replace('___', ' - ').replace('_', ' ')}",
            "organic": ["Remove affected leaves immediately", "Apply copper-based fungicide", "Improve air circulation", "Avoid overhead watering"],
            "chemical": ["Consult local agricultural extension office", "Broad-spectrum fungicide may help"],
            "prevention": ["Crop rotation", "Use disease-free seeds", "Proper plant spacing", "Regular monitoring"]
        }, urgency

def main():
    st.title("🌿 Crop Leaf Disease Detection System")
    st.markdown("### AI-Powered Disease Diagnosis with Severity Analysis")
    
    try:
        model, scaler, class_names = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔬 Analyze Leaf", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    features = extract_all_features(img_array)
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prediction_idx = model.predict(features_scaled)[0]
                    prediction = class_names[prediction_idx]
                    
                    severity_score, severity_class, severity_color, diseased_mask = calculate_severity_score(img_array)
                    orig, _, mask, _ = preprocess_image(img_array)
                    
                    st.markdown("---")
                    st.subheader("📊 Diagnosis Results")
                    
                    result_col1, result_col2 = st.columns([2, 1])
                    
                    with result_col1:
                        formatted_prediction = format_class_name(prediction)
                        if "healthy" in prediction.lower():
                            st.success(f"### ✅ {formatted_prediction}")
                            st.balloons()
                        else:
                            st.warning(f"### ⚠️ {formatted_prediction}")
                    
                    with result_col2:
                        if severity_color == "green":
                            st.success(f"### 🟢 {severity_class}")
                        elif severity_color == "yellow":
                            st.warning(f"### 🟡 {severity_class}")
                        elif severity_color == "orange":
                            st.warning(f"### 🟠 {severity_class}")
                        else:
                            st.error(f"### 🔴 {severity_class}")
                        st.metric("Severity", f"{severity_score:.1f}%")
                    
                    st.markdown("---")
                    st.subheader("📈 Severity Visualization")
                    
                    viz_col1, viz_col2, viz_col3 = st.columns([1, 1, 1])
                    
                    with viz_col1:
                        st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                    
                    with viz_col2:
                        if diseased_mask is not None:
                            overlay = orig.copy()
                            overlay[diseased_mask > 0] = [0, 0, 255]
                            result_img = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
                            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Affected Regions", use_container_width=True)
                    
                    with viz_col3:
                        st.markdown("#### Severity Scale")
                        st.progress(min(severity_score / 100, 1.0))
                        st.markdown(f"**{severity_score:.1f}%** affected")
                    
                    st.markdown("---")
                    st.subheader("💊 Treatment Recommendations")
                    
                    treatment, urgency = get_treatment_recommendation(prediction, severity_class)
                    
                    st.info(f"**{urgency}**")
                    st.markdown(f"*{treatment['description']}*")
                    
                    treat_col1, treat_col2, treat_col3 = st.columns(3)
                    
                    with treat_col1:
                        st.markdown("#### 🌱 Organic Treatment")
                        for item in treatment['organic']:
                            st.markdown(f"• {item}")
                    
                    with treat_col2:
                        st.markdown("#### 🧪 Chemical Treatment")
                        for item in treatment['chemical']:
                            st.markdown(f"• {item}")
                    
                    with treat_col3:
                        st.markdown("#### 🛡️ Prevention")
                        for item in treatment['prevention']:
                            st.markdown(f"• {item}")
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
        st.markdown("## 🌿 About")
        st.markdown("AI-powered crop disease detection using traditional ML")
        
        st.markdown("---")
        st.markdown("### 📊 Supported Crops")
        st.markdown("🍅 Tomato")
        st.markdown("🥔 Potato")
        st.markdown("🌽 Corn/Maize")
        
        st.markdown("---")
        st.markdown("### ✨ Features")
        st.markdown("📈 Severity Scoring")
        st.markdown("💊 Treatment Recommendations")
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developed By")
        st.markdown("**Muhammad Haris** (413826)")
        st.markdown("**Muhammad Arham Siddiqui** (428887)")
        st.markdown("")
        st.markdown("**Course:** CS-471 Machine Learning")
        st.markdown("**Class:** BEE-14 B, NUST")

if __name__ == "__main__":
    main()
