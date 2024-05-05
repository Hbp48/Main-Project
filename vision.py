import os
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
import pyttsx3
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import torch
import pytesseract
from super_gradients.training import models
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
import threading

load_dotenv()

# Set up generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro-vision")

# Define the absolute path to the YOLO-NAS checkpoint
checkpoint_path = os.path.abspath("models/yolo-nas/yolo_nas_s.pt")

# Initialize text-to-speech engine and lock
engine_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)

def get_gemini_response(input_text, image):
    if input_text:
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)

    if response is None:
        return ""
    return response.text

def speak(text):
    with engine_lock:
        engine = pyttsx3.init()
        engine.setProperty("rate", 200)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()

def image_detection(image):
    # Load the YOLO-NAS-s model directly from the checkpoint
    speak("Loading YOLO-NAS-s model...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    speak("YOLO-NAS-s model loaded.")

    # Perform inference
    speak("Performing inference on image...")
    results = model(image)  # Run inference with the YOLO-NAS-s model
    speak("Inference complete.")

    # Process the inference result
    detected_objects = []
    for detection in results.xyxy[0]:
        class_index = int(detection[5])
        confidence = float(detection[4])
        bbox = detection[:4]

        class_name = results.names[class_index]
        x1, y1, x2, y2 = bbox

        detected_objects.append((class_name, confidence, (x1, y1, x2, y2)))

    return detected_objects

# Main Streamlit application code
st.set_page_config(page_title="Object Narration")
st.header("Welcome to Object Narration")

# Speak the welcome message
speak("Welcome to Object Narration")

upload_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if upload_file:
    image = Image.open(upload_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    speak("Analyzing the image...")
    detected_objects = image_detection(image)
    if detected_objects:
        objects_text = ", ".join([obj[0] for obj in detected_objects])
        speak(f"I detected {objects_text}")
        st.subheader("The Object detection Response is")
        st.write(objects_text)

    # OCR
    ocr_text = pytesseract.image_to_string(image)
    if ocr_text:
        speak("I found some text in the image")
        st.subheader("The OCR Response is")
        st.write(ocr_text)
        speak(ocr_text)

    # Image captioning
    speak("Captioning the image...")
    caption = get_gemini_response("", image)
    if caption:
        speak("The caption for the image is")
        st.subheader("The Caption Response is")
        st.write(caption)
        speak(caption)

    # Generate narration based on detected objects, OCR text, and image caption
    narration_input = " ".join([obj[0] for obj in detected_objects]) + " " + ocr_text + " " + caption
    narration = get_gemini_response(narration_input, image)
    if narration:
        speak("Based on the analysis, here is the narration of the image")
        st.subheader("The Narration Response is")
        st.write(narration)
        speak(narration)

    speak("Streamlit application completed.")
