import cv2
import streamlit as st
import pyttsx3
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import torch
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import threading
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Set up generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize text-to-speech engine and lock
engine_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)

def get_gemini_response(input_text, image):
    response = model.generate_content([input_text, image]) if input_text else model.generate_content(image)
    return response.text if response else ""

def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty("rate", 200)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    executor.submit(_speak)

def provide_audio_guidance(step, text=None):
    if step == "welcome":
        speak("Welcome to Accesify. Please upload an image to get started.")
    elif step == "image_uploaded":
        speak("Image uploaded. Analyzing the image.")
    elif step == "object_detection":
        speak("Object detection complete. Detected objects are:")
    elif step == "ocr":
        speak("Text extraction complete. The extracted text is:")
    elif step == "image_captioning":
        speak("Image captioning complete. The caption for the image is:")
    elif step == "narration":
        speak("Generating narration based on the analysis.")
    elif step == "narration_complete":
        speak("Narration complete.")
    elif step == "question":
        speak("You can now ask a question about the image. Enter your question and click the 'Ask the question' button.")
    elif step == "question_response":
        speak(f"The response is: {text}")
    elif step == "exit":
        speak("Thank you for using Object Narration. Goodbye.")

def image_detection(image):
    # Set up Detectron2 object detection
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # Perform object detection
    outputs = predictor(image)
    
    detected_objects = []
    for i, box in enumerate(outputs.pred_boxes):
        class_index = int(outputs.pred_classes[i])
        confidence = float(outputs.scores[i])
        bbox = [int(coord) for coord in box]

        class_name = cfg.DATASETS.COCO_CLASSES[class_index]
        x1, y1, x2, y2 = bbox

        detected_objects.append((class_name, confidence, (x1, y1, x2, y2)))

    return detected_objects

def img_cap(image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Main Streamlit application code
st.set_page_config(page_title="Object Narration")
st.header("Welcome to Accesify")

provide_audio_guidance("welcome")

# Create a video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video
    ret, frame = cap.read()

    # Perform object detection
    detected_objects = image_detection(frame)
    if detected_objects:
        objects_text = ", ".join([obj[0] for obj in detected_objects])
        provide_audio_guidance("object_detection")
        speak(f"I detected {objects_text}")
        st.subheader("The Object detection Response is")
        st.write(objects_text)

    # Perform OCR
    ocr_text = pytesseract.image_to_string(frame)
    if ocr_text:
        provide_audio_guidance("ocr")
        st.subheader("The OCR Response is")
        st.write(ocr_text)
        speak(ocr_text)

    # Perform image captioning
    provide_audio_guidance("image_captioning")
    caption = img_cap(frame)
    if caption:
        st.subheader("The Caption Response is")
        st.write(caption)
        speak(caption)

    # Perform narration
    provide_audio_guidance("narration")
    narration_input = "Object detected: ".join([obj[0] for obj in detected_objects]) + "  OCR: " + ocr_text + " Image caption: " + caption + " generate a narration"
    narration = get_gemini_response(narration_input, frame)
    if narration:
        st.subheader("The Narration Response is")
        st.write(narration)
        speak("Based on the analysis, here is the narration of the image is : " + narration)

    # Display the frame in the Streamlit app
    st.image(frame, channels="BGR", use_column_width=True)

    # Add a way for the user to exit the application
    if st.button("Stop"):
        break

provide_audio_guidance("exit")
cap.release()