\# Deepfake Detection Project



This project uses an EfficientNet-B0 model to detect deepfakes in images.

The application is served as a web app using FastAPI for the backend API

and Streamlit for the frontend.



\## 🚀 How to Run



1\.  Clone this repository:

&nbsp;   `git clone https://github.com/YOUR-USERNAME/YOUR-PROJECT-NAME.git`



2\.  Install the required packages:

&nbsp;   `pip install -r requirements.txt`



3\.  \*\*Download the Model:\*\*

&nbsp;   This repository does \*\*not\*\* include the trained model file. You must

&nbsp;   provide your own `baseline\_best.pth` file.



4\.  \*\*Place the Model:\*\*

&nbsp;   Create a folder path `models/checkpoints/` and place your

&nbsp;   `baseline\_best.pth` file inside it.



5\.  \*\*Run the API:\*\*

&nbsp;   `python -m uvicorn deepfake\_api.main:app --reload`



6\.  \*\*Run the Web App (in a new terminal):\*\*

&nbsp;   `streamlit run app.py`

