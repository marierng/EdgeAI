# Age, Gender, and Expression Detection App

This is a Streamlit application for detecting age, gender, and facial expressions in images using pre-trained models. 
The application allows users to either upload an image or use their webcam to perform the detection.

## Features

- **Face Detection**: Detects faces in the image using OpenCV's DNN module.
- **Age Estimation**: Estimates the age of detected faces.
- **Gender Classification**: Classifies the gender of detected faces.
- **Expression Recognition**: Recognizes facial expressions from detected faces.

## Requirements

- Python 3.x
- Streamlit
- OpenCV
- TensorFlow
- PIL (Pillow)
- NumPy

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/age-gender-expression-detection.git
   cd age-gender-expression-detection

## Install the Requirements

## Download the pre-trained models and place them in the project directory: Download them here: 
https://drive.google.com/drive/folders/1jSVUEGa-i96karCKsNIH-cApKxo-58YV 

opencv_face_detector_uint8.pb
opencv_face_detector.pbtxt
age_net.caffemodel
age_deploy.prototxt
gender_net.caffemodel
gender_deploy.prototxt
model.h5 (for expression recognition)

## Usage
Run the Streamlit app:


    streamlit run app.py

## Application Structure

**Load Models:** The application loads pre-trained models for face detection, age estimation, gender classification, and expression recognition.

**Detect and Predict:** The main function that processes the image or video frames to detect faces and predict age, gender, and expression.

**Process Image:** A helper function to process the uploaded image and draw bounding boxes around detected faces with labels.

**Streamlit UI:** The user interface created with Streamlit, allowing users to choose between uploading an image or using the webcam.

## How It Works
**Load Models:** Models are loaded using OpenCV and TensorFlow.

**Detect Faces:** Faces are detected in the input image or video frames.

**Predict Attributes:** For each detected face, the age, gender, and expression are predicted.

**Display Results:** The results are displayed on the image or video frame with bounding boxes and labels.

## Acknowledgements
This application uses pre-trained models for face detection, age estimation, gender classification, and expression recognition. Special thanks to the developers of these models.
