import streamlit as st
import requests
import io
from PIL import Image

# --- Page Setup ---
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🤖",
    layout="wide"
)

# --- App ---
st.title("Deepfake Detection Project 🤖")

st.write("Upload an image or video, and the model will analyze if it's a deepfake.")

# URL of your running FastAPI backend
API_URL = "http://127.0.0.1:8000/predict"

# 1. Create the file uploader
uploaded_file = st.file_uploader("Choose a file (image or video)...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # Display the uploaded file
    st.write("**Your Uploaded File:**")
    # Check if it's an image or video to display it properly
    file_type = uploaded_file.type
    if "image" in file_type:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    elif "video" in file_type:
        st.video(uploaded_file)
        
    # 2. Add a button to run detection
    if st.button("Detect Deepfake"):
        
        # 3. When button is clicked, show a "loading" spinner
        with st.spinner("Analyzing... Please wait."):
            
            # 4. Send the file to the FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    # 5. Get the result and display it
                    result = response.json()
                    
                    st.write("---")
                    st.subheader("🔬 Analysis Result")
                    
                    prediction = result.get("prediction")
                    confidence = result.get("confidence")
                    
                    if prediction == "FAKE":
                        st.error(f"**Prediction: FAKE** (Confidence: {confidence*100:.2f}%)")
                    else:
                        st.success(f"**Prediction: REAL** (Confidence: {confidence*100:.2f}%)")
                        
                    st.write("---")
                    st.json(result) # Show the full JSON response for debugging
                
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Error: Could not connect to the API. Is it running?")