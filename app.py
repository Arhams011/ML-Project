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
        "early blight": {
            "description": "Fungal disease causing dark spots with concentric rings",
            "organic": ["Remove infected leaves", "Apply copper fungicide", "Improve air circulation"],
            "chemical": ["Chlorothalonil", "Mancozeb"],
            "prevention": ["Crop rotation", "Mulching", "Drip irrigation"]
        },
        "late blight": {
            "description": "Serious fungal disease causing water-soaked lesions",
            "organic": ["Remove infected plants immediately", "Copper-based fungicides"],
            "chemical": ["Metalaxyl", "Chlorothalonil"],
            "prevention": ["Avoid overhead watering", "Plant resistant varieties", "Good spacing"]
        },
        "bacterial spot": {
            "description": "Bacterial infection causing small dark spots",
            "organic": ["Copper spray", "Remove infected leaves", "Improve drainage"],
            "chemical": ["Copper hydroxide", "Streptomycin (if approved)"],
            "prevention": ["Use disease-free seeds", "Avoid overhead irrigation", "Sanitize tools"]
        },
        "septoria leaf spot": {
            "description": "Fungal disease with circular spots with gray centers",
            "organic": ["Neem oil", "Remove lower leaves", "Mulch around plants"],
            "chemical": ["Chlorothalonil", "Mancozeb"],
            "prevention": ["Stake plants", "Water at base", "Remove debris"]
        },
        "leaf mold": {
            "description": "Fungal disease with pale green spots on upper leaves",
            "organic": ["Improve ventilation", "Reduce humidity", "Sulfur spray"],
            "chemical": ["Chlorothalonil"],
            "prevention": ["Space plants properly", "Prune for air flow", "Control humidity"]
        },
        "common rust": {
            "description": "Fungal disease with small circular to elongated pustules",
            "organic": ["Sulfur-based fungicides", "Remove infected leaves early"],
            "chemical": ["Azoxystrobin", "Propiconazole"],
            "prevention": ["Plant resistant varieties", "Early planting", "Proper spacing"]
        },
        "northern leaf blight": {
            "description": "Long gray-green lesions on leaves",
            "organic": ["Rotate crops", "Remove crop residue"],
            "chemical": ["Pyraclostrobin", "Azoxystrobin"],
            "prevention": ["Use resistant hybrids", "Plow under residue", "Crop rotation"]
        },
        "gray leaf spot": {
            "description": "Rectangular lesions with gray coloration",
            "organic": ["Crop rotation", "Tillage to bury residue"],
            "chemical": ["Strobilurin fungicides"],
            "prevention": ["Resistant varieties", "Residue management", "Crop rotation"]
        }
    }
    
    for key in treatments:
        if key in disease_lower:
            treatment = treatments[key]
            urgency = "🔴 URGENT" if severity_class == "Severe" else "🟠 IMPORTANT" if severity_class == "Moderate" else "🟡 MONITOR"
            return treatment, urgency
    
    return None, "ℹ️ MONITOR"

def main():
    st.title("🌿 Crop Leaf Disease Detection System")
    st.markdown("### AI-Powered Disease Diagnosis with Severity Analysis")
    st.markdown("Upload a leaf image to detect diseases in **Tomato**, **Potato**, or **Corn/Maize** plants")
    
    try:
        model, scaler, class_names = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please ensure model files are present in the app directory")
        return
    
    uploaded_file = st.file_uploader("📤 Choose a leaf image", type=["jpg", "jpeg", "png"], help="Upload a clear image of a leaf")
    
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
            with st.spinner("🔍 Analyzing image..."):
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
                        st.metric("Severity", f"{severity_score:.1f}%", help="Percentage of leaf affected by disease")
                    
                    st.markdown("---")
                    st.subheader("📈 Severity Analysis & Visualization")
                    
                    viz_col1, viz_col2, viz_col3 = st.columns([1, 1, 1])
                    
                    with viz_col1:
                        st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original Leaf", use_container_width=True)
                    
                    with viz_col2:
                        if diseased_mask is not None:
                            overlay = orig.copy()
                            overlay[diseased_mask > 0] = [0, 0, 255]
                            result_img = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
                            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Diseased Regions (Red)", use_container_width=True)
                    
                    with viz_col3:
                        st.markdown("#### Severity Scale")
                        st.progress(min(severity_score / 100, 1.0))
                        st.markdown(f"**{severity_score:.1f}%** of leaf affected")
                        st.markdown("---")
                        if severity_score < 5:
                            st.success("🟢 **Healthy/Minimal**\n\nNo immediate action needed")
                        elif severity_score < 15:
                            st.warning("🟡 **Mild**\n\nMonitor closely and consider preventive treatment")
                        elif severity_score < 35:
                            st.warning("🟠 **Moderate**\n\nTreatment recommended soon")
                        else:
                            st.error("🔴 **Severe**\n\n⚠️ Urgent treatment required!")
                    
                    if "healthy" not in prediction.lower():
                        st.markdown("---")
                        st.subheader("💊 Treatment Recommendations")
                        
                        treatment, urgency = get_treatment_recommendation(prediction, severity_class)
                        
                        if treatment:
                            st.info(f"**{urgency}** - {treatment['description']}")
                            
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
                        else:
                            st.info("Consult with a local agricultural expert for specific treatment recommendations.")
                    
                    st.markdown("---")
                    st.success("✅ Analysis complete! Download this page as PDF for your records.")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("Please try uploading a different image or check image quality")
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
        st.markdown("## 🌿 About")
        st.markdown("This application uses **Traditional Machine Learning** with handcrafted features:")
        st.markdown("• **Color** - HSV histograms")
        st.markdown("• **Texture** - GLCM features")
        st.markdown("• **Shape** - Morphological features")
        
        st.markdown("---")
        st.markdown("### 📊 Supported Crops")
        st.markdown("🍅 **Tomato** - 3 disease types")
        st.markdown("🥔 **Potato** - 3 disease types")
        
        st.markdown("---")
        st.markdown("### ✨ Novel Features")
        st.markdown("📈 **Severity Scoring**")
        st.markdown("Quantifies disease as % of affected leaf area")
        st.markdown("")
        st.markdown("💊 **Treatment Recommendations**")
        st.markdown("Organic & chemical treatment options")
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developed By")
        st.markdown("**Muhammad Haris** (413826)")
        st.markdown("**Muhammad Arham Siddiqui** (428887)")
        st.markdown("")
        st.markdown("**Course:** CS-471 Machine Learning")
        st.markdown("**Institution:** NUST, Pakistan")
        st.markdown("**Class:** BEE-14 B")
        
        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        st.markdown("1. Upload a clear leaf image")
        st.markdown("2. Click 'Analyze Leaf' button")
        st.markdown("3. View diagnosis and severity")
        st.markdown("4. Follow treatment recommendations")
        
        st.markdown("---")
        st.caption("© 2024 NUST | All rights reserved")

if __name__ == "__main__":
    main()
