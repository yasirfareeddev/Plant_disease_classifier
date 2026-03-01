import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# LOAD MODEL AND DATA
# =====================================================
@st.cache_resource
def load_model_and_data():
    """Load model, class indices, and treatments (cached for performance)"""
    working_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load model
    model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
    model = tf.keras.models.load_model(model_path)
    
    # Load class indices
    class_indices_path = os.path.join(working_dir, "class_indices.json")
    class_indices = json.load(open(class_indices_path))
    
    # Load treatments
    treatments_path = os.path.join(working_dir, "treatments.json")
    treatments = json.load(open(treatments_path))
    
    return model, class_indices, treatments


model, class_indices, treatments = load_model_and_data()


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def load_and_preprocess_image(image, target_size=(224, 224)):
    """Load and preprocess image for prediction"""
    # Convert to RGB (removes alpha/transparency channel from PNG)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image
    img = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Scale to [0, 1]
    img_array = img_array.astype('float32') / 255.
    
    return img_array


def predict_image_class(model, image, class_indices):
    """Predict disease class and return prediction with confidence"""
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence


def get_treatment_info(disease_name, treatments):
    """Get treatment information for a disease"""
    if disease_name in treatments:
        return treatments[disease_name]
    return {
        "treatment": "Consult agricultural expert",
        "medicine": "N/A",
        "suggestion": "No data available for this disease"
    }


def is_healthy(disease_name):
    """Check if the plant is healthy"""
    return "healthy" in disease_name.lower()


# =====================================================
# SIDEBAR - INFORMATION
# =====================================================
with st.sidebar:
    st.header("🌱 About This App")
    st.markdown("""
    This app uses a **Convolutional Neural Network (CNN)** 
    to detect plant diseases from leaf images.
    
    **Supported Plants:**
    - 🍎 Apple (4 classes)
    - 🌽 Corn/Maize (4 classes)
    - 🫑 Pepper Bell (2 classes)
    - 🥔 Potato (3 classes)
    - 🍅 Tomato (10 classes)
    
    **Total Classes:** 23
    """)
    
    st.divider()
    
    st.header("📊 Model Info")
    st.info("""
    - **Architecture:** CNN with GlobalAveragePooling2D
    - **Input Size:** 224x224 pixels
    - **Training Accuracy:** ~88%
    - **Model Training Time:** 161m       
    """)
    
    st.divider()
    
    st.header("⚠️ Disclaimer")
    st.warning("""
    This app provides **suggestions only**. 
    For severe infections, consult a local 
    agricultural expert or plant pathologist.
    """)


# =====================================================
# MAIN APP
# =====================================================
st.title("🌿 Plant Disease Classifier & Treatment Advisor")
st.markdown("""
Upload an image of a plant leaf to detect diseases and get **personalized treatment recommendations**.
""")

# File uploader
uploaded_image = st.file_uploader(
    "📤 Upload a leaf image...", 
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

# If image is uploaded
if uploaded_image is not None:
    # Display image
    image = Image.open(uploaded_image)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True, caption="Your uploaded leaf image")
    
    with col2:
        st.subheader("🔬 Analysis Results")
        
        if st.button("🔍 Classify Disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing image... Please wait..."):
                # Get prediction
                prediction, confidence = predict_image_class(model, image, class_indices)
                
                # Get treatment info
                treatment_info = get_treatment_info(prediction, treatments)
                
                # Check if healthy
                healthy = is_healthy(prediction)
                
                # Display prediction
                if healthy:
                    st.success(f"✅ **Prediction:** {prediction}")
                    st.info(f"📊 **Confidence:** {confidence:.2f}%")
                    
                    # Healthy plant care tips
                    st.markdown("### 🌱 Plant Care Tips")
                    st.success(f"""
                    **Status:** Your plant is **HEALTHY**!
                    
                    **Recommendation:** {treatment_info['suggestion']}
                    
                    **General Care:**
                    - 💧 Water regularly according to plant type
                    - ☀️ Ensure adequate sunlight
                    - 🌡️ Monitor temperature and humidity
                    - 🧪 Use balanced fertilizer
                    - 🔍 Inspect regularly for early disease signs
                    """)
                else:
                    st.error(f"⚠️ **Prediction:** {prediction}")
                    st.info(f"📊 **Confidence:** {confidence:.2f}%")
                    
                    # Disease treatment information
                    st.markdown("### 🏥 Treatment Recommendations")
                    
                    st.warning(f"""
                    **🩺 Disease Detected:** {prediction.replace('_', ' ')}
                    
                    **📋 Treatment:** {treatment_info['treatment']}
                    
                    **💊 Recommended Medicine:** {treatment_info['medicine']}
                    
                    **💡 Suggestion:** {treatment_info['suggestion']}
                    """)
                    
                    # Additional care tips for diseased plants
                    st.markdown("### ⚠️ Immediate Actions")
                    st.error("""
                    1. **Isolate** the affected plant immediately
                    2. **Remove** and destroy infected leaves/plants
                    3. **Disinfect** tools after use
                    4. **Avoid** overhead watering
                    5. **Improve** air circulation
                    6. **Monitor** nearby plants for spread
                    """)
                    
                    # Warning box
                    st.warning("""
                    **⚠️ Important:** If the infection is severe or spreading rapidly, 
                    please consult a local agricultural expert or plant pathologist 
                    for professional diagnosis and treatment.
                    """)
    
    # Additional information section
    st.divider()
    st.markdown("### 📚 Additional Resources")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.info("""
        **📖 Prevention Tips**
        - Practice crop rotation
        - Use disease-resistant varieties
        - Maintain proper plant spacing
        """)
    
    with col_b:
        st.info("""
        **🔧 Best Practices**
        - Regular field inspection
        - Proper sanitation
        - Balanced fertilization
        """)
    
    with col_c:
        st.info("""
        **🌍 Environmental Care**
        - Optimal irrigation
        - Good air circulation
        - Proper drainage
        """)

else:
    # Show placeholder when no image uploaded
    st.info("⬆️  Please upload a plant leaf image to get started!")
    
    # Example images section
    st.markdown("### 📸 Example Images")
    st.markdown("""
    For best results, ensure your image:
    - ✅ Shows a clear view of the leaf
    - ✅ Has good lighting
    - ✅ Focuses on affected areas
    - ✅ Is in JPG, JPEG, or PNG format
    """)


# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built by Yasir Fareed using Streamlit & TensorFlow | Plant Disease Classifier v1.0</p>
</div>
""", unsafe_allow_html=True)