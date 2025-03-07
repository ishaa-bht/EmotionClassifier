import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import base64

# Cache resources to improve performance
@st.cache_resource
def load_model_and_resources():
    try:
        # Load class labels and preprocessing config
        class_names = joblib.load("class_names.joblib")
        transform_config = joblib.load("transform_config.joblib")
        
        # Define the ResNet-50 model
        class EmotionClassifier(nn.Module):
            def __init__(self, num_classes):
                super(EmotionClassifier, self).__init__()
                self.resnet = models.resnet50(weights=None)
                
                # Modify the final layers
                num_features = self.resnet.fc.in_features
                self.resnet.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )
                
                # Add dropout in intermediate layers
                self.resnet.layer3.add_module('dropout', nn.Dropout(0.2))
                self.resnet.layer4.add_module('dropout', nn.Dropout(0.2))

            def forward(self, x):
                return self.resnet(x)

        # Load the trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(class_names)
        model = EmotionClassifier(num_classes)
        model.load_state_dict(torch.load("model_state_dict.pth", map_location=device))
        model.to(device)
        model.eval()
        
        return model, class_names, transform_config, device
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        return None, None, None, None

# Image preprocessing
def preprocess_image(image, transform_config, device):
    transform = transforms.Compose([
        transforms.Resize(transform_config["resize_size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            transform_config["normalization_mean"],
            transform_config["normalization_std"]
        )
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    return img_tensor

# Create a bar chart for emotion probabilities
def create_probability_chart(emotion_probs):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Sort by probability
    sorted_data = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
    emotions = [x[0] for x in sorted_data]
    probs = [x[1] for x in sorted_data]
    
    # Create bars with colors (highlight the highest probability)
    colors = ['#1f77b4'] * len(emotions)
    colors[0] = '#ff7f0e'  # Highlight the highest probability
    
    ax.barh(emotions, probs, color=colors)
    ax.set_xlabel('Probability')
    ax.set_title('Emotion Probability Distribution')
    ax.set_xlim(0, 1)
    
    # Add values on bars
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Function to get sample images
def get_sample_images():
    # In a real app, you would include these files in your project
    # This is just a placeholder - you'd need to add your own sample images
    sample_images = {
        "Happy Woman": "samples/happy_woman.jpg",
        "Angry Man": "samples/angry_man.jpg", 
        "Surprised Child": "samples/surprised_child.jpg",
        "Sad Elderly": "samples/sad_elderly.jpg"
    }
    return sample_images

# Set page configuration
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😀",
    layout="centered"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 1rem;
        text-align: center;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #616161;
        font-size: 0.8rem;
    }
    .stProgress .st-emotion-cache-1c7u610 {
        background-color: #1E88E5;
    }
    .container {
        max-width: 800px;
        margin: auto;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.markdown('<p class="main-header">Facial Emotion Classifier</p>', unsafe_allow_html=True)

# Load model and resources
model, class_names, transform_config, device = load_model_and_resources()

if None in (model, class_names, transform_config, device):
    st.error("Failed to load necessary resources. Please check that all model files are available.")
    st.stop()

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Upload Image", "Try with Samples", "Camera Capture"])

# Function to classify emotion
def classify_emotion(image):
    progress_bar = st.progress(0)
    
    # Update progress
    progress_bar.progress(25)
    st.write("Preprocessing image...")
    
    img_tensor = preprocess_image(image, transform_config, device)
    
    # Update progress
    progress_bar.progress(50)
    st.write("Running model inference...")
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Update progress
    progress_bar.progress(75)
    st.write("Processing results...")
    
    confidence, predicted = torch.max(probabilities, 0)
    predicted_class = class_names[predicted.item()]
    
    # Create dictionary of emotions and probabilities
    emotion_probs = {emotion: prob.item() for emotion, prob in zip(class_names, probabilities)}
    
    # Update progress
    progress_bar.progress(100)
    st.write("Done!")
    
    # Return results
    return predicted_class, confidence.item(), emotion_probs

# Tab 1: Upload Image
with tab1:
    st.markdown('<p class="sub-header">Upload an image to classify the emotion</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Classify Emotion", key="classify_upload"):
            with st.spinner("Processing..."):
                try:
                    predicted_class, confidence, emotion_probs = classify_emotion(image)
                    
                    # Display results in a nice format
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown(f"### Predicted Emotion: **{predicted_class}**")
                        st.markdown(f"### Confidence: **{confidence:.2f}**")
                        
                        # Create gauge for confidence
                        st.progress(confidence)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        # Create and display probability chart
                        chart_buf = create_probability_chart(emotion_probs)
                        st.image(chart_buf, caption="Emotion Probabilities", use_container_width=True)
                    
                    # Create a dataframe for detailed probabilities
                    prob_df = pd.DataFrame({
                        'Emotion': list(emotion_probs.keys()),
                        'Probability': list(emotion_probs.values())
                    }).sort_values('Probability', ascending=False)
                    
                    with st.expander("View Detailed Probabilities"):
                        st.dataframe(prob_df, use_container_width=True)
                    
                    # Add option to download results
                    csv = prob_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="emotion_results.csv">Download results as CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error processing image: {e}")

# Tab 2: Sample Images
with tab2:
    st.markdown('<p class="sub-header">Try with our sample images</p>', unsafe_allow_html=True)
    
    # Get sample images
    sample_images = get_sample_images()
    
    # Check if sample images are available
    if os.path.exists(next(iter(sample_images.values()), "")):
        # Create a grid of sample images
        cols = st.columns(len(sample_images))
        selected_sample = None
        
        for i, (label, img_path) in enumerate(sample_images.items()):
            with cols[i]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=label, width=150)
                    if st.button("Select", key=f"sample_{i}"):
                        selected_sample = img_path
        
        if selected_sample:
            image = Image.open(selected_sample).convert("RGB")
            st.image(image, caption="Selected Sample Image", width=300)
            
            if st.button("Classify Emotion", key="classify_sample"):
                with st.spinner("Processing..."):
                    try:
                        predicted_class, confidence, emotion_probs = classify_emotion(image)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.markdown(f"### Predicted Emotion: **{predicted_class}**")
                            st.markdown(f"### Confidence: **{confidence:.2f}**")
                            st.progress(confidence)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            chart_buf = create_probability_chart(emotion_probs)
                            st.image(chart_buf, caption="Emotion Probabilities", use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
    else:
        st.warning("Sample images not found. Please upload your own image in the Upload tab.")

# Tab 3: Camera Capture
with tab3:
    st.markdown('<p class="sub-header">Capture an image with your camera</p>', unsafe_allow_html=True)
    
    # Camera input
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        
        if st.button("Classify Emotion", key="classify_camera"):
            with st.spinner("Processing..."):
                try:
                    predicted_class, confidence, emotion_probs = classify_emotion(image)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown(f"### Predicted Emotion: **{predicted_class}**")
                        st.markdown(f"### Confidence: **{confidence:.2f}**")
                        st.progress(confidence)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        chart_buf = create_probability_chart(emotion_probs)
                        st.image(chart_buf, caption="Emotion Probabilities", use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error processing image: {e}")

# Add a section for emotion explanations
with st.expander("Emotion Descriptions and Context"):
    st.write("""
    ### Understanding Facial Emotions
    
    - **Happy**: Characterized by smiling, raised cheeks, and crinkled eyes. Associated with positive experiences and satisfaction.
    
    - **Sad**: Features drooping eyelids, downturned mouth, and sometimes furrowed brows. Related to disappointment, loss, and grief.
    
    - **Angry**: Identified by lowered brows, intense eyes, and compressed lips. Indicates frustration, hostility, or irritation.
    
    - **Surprised**: Shows raised eyebrows, widened eyes, and often an open mouth. A reaction to unexpected events.
    
    - **Neutral**: Exhibits minimal muscle activity with a relaxed face. The baseline expression without strong emotion.
    
    - **Fear**: Distinguished by raised eyebrows, widened eyes, and often an open mouth. Associated with threat perception and anxiety.
    
    - **Disgust**: Shown through a wrinkled nose, raised upper lip, and lowered brows. A response to unpleasant stimuli.
    """)

# Add history section (demonstration only - in a real app this would store past results)
with st.expander("History (Session Only)"):
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if st.session_state.history:
        for i, (timestamp, pred_class, conf) in enumerate(st.session_state.history):
            st.write(f"{i+1}. {timestamp}: {pred_class} (Confidence: {conf:.2f})")
    else:
        st.write("No history available yet. Process an image to record results.")

# Footer
st.markdown('<p class="footer">Facial Emotion Recognition System • Built with Streamlit and PyTorch</p>', unsafe_allow_html=True)