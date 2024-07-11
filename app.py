import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

# Function to load OpenCV models
@st.cache_resource
def load_opencv_model(model_path, config_path):
    return cv2.dnn.readNet(model_path, config_path)

# Load models
@st.cache_resource
def load_models():
    face_net = load_opencv_model("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
    age_net = load_opencv_model("age_net.caffemodel", "age_deploy.prototxt")
    gender_net = load_opencv_model("gender_net.caffemodel", "gender_deploy.prototxt")
    expression_model = tf.keras.models.load_model("model.h5", compile=False)
    return face_net, age_net, gender_net, expression_model

face_net, age_net, gender_net, expression_model = load_models()

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
gender_list = ['Male', "Female"]
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def detect_and_predict(frame, face_net, expression_model, gender_net, age_net):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    face_net.setInput(blob)
    detections = face_net.forward()
    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # Expression Detection
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray_face, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            expression_preds = expression_model.predict(roi)[0]
            expression_label = emotion_dict[expression_preds.argmax()]

            # Gender and Age Detection
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            results.append((x1, y1, x2, y2, expression_label, gender, age))

    return results

def process_image(image):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = detect_and_predict(frame, face_net, expression_model, gender_net, age_net)

    for (x1, y1, x2, y2, expression, gender, age) in results:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{expression}, {gender}, {age}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

st.title("Age, Gender, and Expression Detection")

option = st.radio("Choose input method:", ('Upload Image', 'Use Webcam'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Detect'):
            result_image = process_image(image)
            st.image(result_image, caption='Processed Image', use_column_width=True)

elif option == 'Use Webcam':
    st.write("Webcam feed:")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detect_and_predict(frame, face_net, expression_model, gender_net, age_net)

        for (x1, y1, x2, y2, expression, gender, age) in results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{expression}, {gender}, {age}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

    camera.release()

st.write("Note: This app uses pre-trained models for face detection, age estimation, gender classification, and expression recognition.")