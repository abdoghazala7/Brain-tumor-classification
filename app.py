import streamlit as st
from PIL import Image
import torch
import pandas as pd

# Import configuration and utility functions from other files
import config
import model_loader
import image_utils
import predictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Polished Look ---
st.markdown("""
<style>
/* Hide Streamlit's default components */
[data-testid="stFileUploaderFile"] {
    display: none;
}
.st-emotion-cache-1v0mbdj > img {
    font-style: italic;
    color: #6c757d;
}
/* Custom containers for results */
.result-container {
    padding: 20px;
    border-radius: 10px;
    background-color: #ffffff;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border: 1px solid #e1e4e8;
    /* Flexbox properties for centering */
    display: flex;
    justify-content: center; /* Horizontal centering */
    align-items: center;     /* Vertical centering */
    min-height: 150px;       /* Give it some height */
}
.result-container h2 {
    color: #0d6efd; /* A nice blue color */
    font-weight: bold;
    margin: 0; /* Remove default margin */
}
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def get_model():
    """Loads and caches the fine-tuned model."""
    with st.spinner('Loading the AI model, please wait...'):
        model = model_loader.load_model()
    return model

model = get_model()

# --- Sidebar Information ---
with st.sidebar:
    st.header("About This Project")
    st.info(
        "This app uses a state-of-the-art deep learning model (`EfficientNetB0`) "
        "fine-tuned with a gradual unfreezing strategy on the Brain Tumor MRI Dataset "
        "to classify brain tumors with **98% accuracy**."
    )
    
    st.markdown("Check out the full project on GitHub:")
    st.markdown(
        "[GitHub Repository](https://github.com/abdoghazala7/Brain-tumor-classification)"
    )

# --- Main App Interface ---
st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload an MRI image of a brain, and our AI will predict the type of tumor "
    "(Glioma, Meningioma, Pituitary) or if there is no tumor. "
    "The confidence scores for each class will then be displayed."
)
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI image...", 
        type=["jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"],
        label_visibility="collapsed"
    )

with col2:
    st.header("📊 Analysis & Results")

    if uploaded_file is None:
        st.info("Your uploaded image and the analysis results will appear here.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_container_width=True)
        
        with st.spinner('🔬 Analyzing the image... Please wait.'):
            try:
                image_bytes = uploaded_file.getvalue()
                image_tensor = image_utils.preprocess_image(image_bytes)
                
                if image_tensor is not None:
                    predicted_class, probabilities = predictor.predict(model, image_tensor)
                    
                    st.markdown("---")
                    st.markdown("The model predicts")
                    st.markdown(
                        f"""
                        <div class="result-container">
                            <h2>{predicted_class.capitalize()}</h2>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.write("")
                    
                    with st.expander("View Confidence Scores"):
                        prob_df = pd.DataFrame({
                            'Class': [name.capitalize() for name in probabilities.keys()],
                            'Probability': [prob for prob in probabilities.values()]
                        }).sort_values(by='Probability', ascending=False)
                        
                        st.dataframe(
                            prob_df.style.format({'Probability': '{:.2%}'}),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.error("Could not preprocess the image. Please try another file.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")