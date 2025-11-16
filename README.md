Deepfake Detection Project

Project Overview

This project is a full-stack web application designed to detect deepfakes in images. It leverages a trained EfficientNet-B0 deep learning model, served via a high-performance Python backend.

The application consists of two main components:

Backend API (FastAPI): A fast and robust API that loads the trained PyTorch model, accepts image uploads, and returns a JSON response with the ("REAL" or "FAKE") prediction and a confidence score.

Frontend (Streamlit): A clean, interactive web interface that allows users to easily upload an image and view the analysis results from the backend.

Technology Stack

Backend: FastAPI

Frontend: Streamlit

Model: PyTorch, EfficientNet-B0

Core Libraries: torchvision, Pillow, uvicorn, requests

Installation and Usage

Follow these steps to set up and run the project locally.

1. Clone the Repository

Clone this repository to your local machine and navigate into the directory:

git clone [https://github.com/Praveen21-06/deepfake-detector.git](https://github.com/Praveen21-06/deepfake-detector.git)
cd deepfake-detector


2. Install Dependencies

Install all the required Python packages using the requirements.txt file:

pip install -r requirements.txt


3. Set Up the Model

This repository does not include the large, trained model file (.pth). You must provide your own.

Create the necessary folders:

mkdir -p models/checkpoints


Place your trained baseline_best.pth file inside this directory. The final path should be:
models/checkpoints/baseline_best.pth

4. Run the Application

The application requires two separate terminals to run the backend and frontend.

Terminal 1: Run the Backend API

Start the FastAPI server:

python -m uvicorn deepfake_api.main:app --reload


The API will be live at http://127.0.0.1:8000.

Terminal 2: Run the Frontend App

In a new terminal, start the Streamlit web application:

streamlit run app.py


Streamlit will automatically open your web browser to the application's URL. You can now upload an image to get a prediction.
