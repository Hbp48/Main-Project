import os
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
import pyttsx3
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import torch
#import pytesseract
#from super_gradients.training import models
#import torchvision.transforms as transforms
#from torchvision.transforms import functional as F
#from pycocotools.coco import COCO
import threading
#import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
#from vertexai.preview import generative_models as genai


load_dotenv()

# Set up generative AI
genai.configure(api_key="AIzaSyAWetxM8whh_s2ISO2QvuabrCQ9Ccq-RV8")
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

from ultralytics import YOLO

def image_detection(image):
    speak("Loading YOLO-NAS-s model for object detection...")
    
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


    speak("YOLO-NAS-s model loaded.")

    speak("Performing inference on image...")
    results = model(image)
    speak("Inference complete.")

    detected_objects = []
    for detection in results.xyxy[0]:
        class_index = int(detection[5])
        confidence = float(detection[4])
        bbox = detection[:4]

        class_name = results.names[class_index]
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

upload_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if upload_file:
    image = Image.open(upload_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)


    provide_audio_guidance("image_uploaded")

    detected_objects = image_detection(image)
    if detected_objects:
        objects_text = ", ".join([obj[0] for obj in detected_objects])
        provide_audio_guidance("object_detection")
        speak(f"I detected {objects_text}")
        st.subheader("The Object detection Response is")
        st.write(objects_text)

    #ocr_text = pytesseract.image_to_string(image)
    #if ocr_text:
     #   provide_audio_guidance("ocr")
      #  st.subheader("The OCR Response is")
       # st.write(ocr_text)
        #speak(ocr_text)

    provide_audio_guidance("image_captioning")
    caption = img_cap(image)
    if caption:
        st.subheader("The Caption Response is")
        st.write(caption)
        speak(caption)

    provide_audio_guidance("narration")
    narration_input = "Object detected: ".join([obj[0] for obj in detected_objects]) + " Image caption: " + caption + " generate a narration"
    narration = get_gemini_response(narration_input, image)
    if narration:
        st.subheader("The Narration Response is")
        st.write(narration)
        speak("Based on the analysis, here is the narration of the image is : " + narration)

    ocr_gemini=get_gemini_response("Do all the ocr of text present in the image", image)
    st.subheader("Text extracted from image")
    st.write(ocr_gemini)
    speak("Text extracted: " + ocr_gemini)
    # index=0
    # t = 1
    # while t != 0:
    #     speak("Enter 0 to exit else 1")
    #    # widget_id = (id for id in range(1, 100_00))
    #     user_input_key = f"user_input{index}"  
    #     user_input = st.text_input("Enter a value", key=user_input_key)
    #     if user_input.isnumeric():
    #         numeric_value = int(user_input)
    #         st.write(f"Numeric value entered: {numeric_value}")
    #     else:
    #         st.write(f"Non-numeric value entered: {user_input}")
    #     if t != 0:
    #         provide_audio_guidance("question")
    #         question_key = f"question_input{index}"  
    #         question = st.text_input("Question:", key=question_key)
    #         submit_key = "submit_button"  
    #         submit = st.button("Ask the question", key=submit_key)
    #         if submit:
    #             response = get_gemini_response(question, image)
    #             provide_audio_guidance("question_response", response)
    #             st.subheader("The Response is")
    #             st.write(response)
    #     index=index+1

    provide_audio_guidance("exit")