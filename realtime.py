import os
import cv2
from dotenv import load_dotenv
import streamlit as st
import pyttsx3
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
import pytesseract
from super_gradients.training import models
import threading

load_dotenv()

# Initialize text-to-speech engine and thread executor
engine_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)

def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty("rate", 200)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    executor.submit(_speak)

def image_detection(frame):
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    results = model(frame)
    detected_objects = []
    for detection in results.xyxy[0]:
        class_index = int(detection[5])
        confidence = float(detection[4])
        x1, y1, x2, y2 = map(int, detection[:4])
        class_name = results.names[class_index]
        detected_objects.append((class_name, confidence, (x1, y1, x2, y2)))
    return detected_objects

def draw_bounding_boxes(frame, detected_objects):
    for obj in detected_objects:
        class_name, confidence, (x1, y1, x2, y2) = obj
        label = f"{class_name} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Main Streamlit application code
st.set_page_config(page_title="Live Object Detection Video Feed")
st.header("Real-Time Object Detection Video Feed")

# Set up the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Could not open the camera.")
else:
    stframe = st.empty()  # Placeholder for displaying frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every few frames to improve real-time performance
        detected_objects = image_detection(frame)

        # Draw bounding boxes on the frame
        frame_with_boxes = draw_bounding_boxes(frame, detected_objects)

        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)

        # Display the current frame in Streamlit
        stframe.image(frame_rgb, caption="Live Camera Feed with Detected Objects", use_column_width=True)

        # Announce detected objects
        if detected_objects:
            objects_text = ", ".join([obj[0] for obj in detected_objects])
            speak(f"I detected {objects_text}")
            st.subheader("Detected Objects")
            st.write(objects_text)

        # Add Stop button to break the loop
        if st.button("Stop"):
            speak("Thank you for using the live object detection feed. Goodbye.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
