import os
import cv2
import threading
import time
from PIL import Image
import pyttsx3
import google.generativeai as genai
import pytesseract
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load environment variables
load_dotenv()

# Set up generative AI with API key
genai.configure(api_key="AIzaSyAWetxM8whh_s2ISO2QvuabrCQ9Ccq-RV8")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize text-to-speech engine and threading lock
engine_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)

# Global variable to hold the captured frame
captured_frame = None

def speak(text):
    """Convert text to speech."""
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty("rate", 200)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    executor.submit(_speak)

def generate_description_for_frame(frame, prompt="Explain this image within 30 words"):
    """Generate a description for a single image frame using the generative model."""
    pil_image = Image.fromarray(frame)
    response = model.generate_content([prompt, pil_image])
    return response.text

def save_description_as_audio(description, filename="description_audio.mp3"):
    """Convert description text to speech and save as an audio file."""
    tts = gTTS(text=description, lang='en')
    tts.save(filename)
    print(f"Audio saved as {filename}")
    return filename

def image_detection(image):
    """Detect objects in an image using YOLO."""
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    results = model(image)
    detected_objects = [(int(detection[5]), float(detection[4]), detection[:4]) for detection in results.xyxy[0]]
    return detected_objects

def img_cap(image):
    """Generate caption for an image."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def capture_frame():
    global captured_frame
    vc = cv2.VideoCapture(0)
    
    if not vc.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        rval, frame = vc.read()
        if not rval:
            break
        
        # Store the captured frame globally
        with engine_lock:
            captured_frame = frame.copy()

        time.sleep(1)  # Capture frames every second

    vc.release()

# Streamlit app interface
st.set_page_config(page_title="Real-Time Object Narration")
st.header("Real-Time Object Narration")

st.write("Live feed will be displayed here with frame processing every 30 seconds.")

# Create a placeholder for the live feed
live_feed_box = st.empty()

# Run OpenCV frame capture in a separate thread
threading.Thread(target=capture_frame, daemon=True).start()

# Streamlit loop to display live feed in a box and trigger frame processing
while True:
    if captured_frame is not None:
        # Display live feed in the box
        live_feed_box.image(captured_frame, caption="Live Video Feed", use_container_width=True)

        # Process frame after every 30 seconds
        current_time = int(time.time())
        if current_time % 30 == 0:  # Process every 30 seconds
            st.write("Processing captured frame...")

            # Object Detection
            detected_objects = image_detection(captured_frame)
            objects_text = ", ".join([str(obj[0]) for obj in detected_objects])
            st.subheader("Detected Objects:")
            st.write(objects_text)
            
            # OCR
            ocr_text = pytesseract.image_to_string(captured_frame)
            st.subheader("Extracted Text:")
            st.write(ocr_text)

            # Image Captioning
            caption = img_cap(captured_frame)
            st.subheader("Image Caption:")
            st.write(caption)

            # Narration Generation
            narration_input = f"Objects: {objects_text}. OCR: {ocr_text}. Caption: {caption}. Provide a narration."
            narration = generate_description_for_frame(captured_frame)
            st.subheader("Generated Narration:")
            st.write(narration)
            speak("Narration: " + narration)

    # Refresh Streamlit UI periodically
    time.sleep(1)  # Refresh UI every 1 second to display live feed
