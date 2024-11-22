import os
import time
import io
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import sounddevice as sd
from gtts import gTTS
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import threading
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load environment variables
load_dotenv()

class ImageDescriptionGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            genai.configure(api_key="AIzaSyBrVHl1xPh5ncCGOApqaK3Lna2XpopBEqY")
            self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
            print("Generative AI configured successfully.")
        except Exception as e:
            print(f"Error configuring generative model: {e}")
            self.model = None

    def generate_description(self, frame: np.ndarray, prompt: str) -> str:
        if not self.model:
            return "Model not available for generating description."
        try:
            pil_image = Image.fromarray(frame)
            response = self.model.generate_content([prompt, pil_image])
            return response.text
        except Exception as e:
            print(f"Error generating description: {e}")
            return "Error generating description."

class TextToSpeech:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)

    def speak(self, text: str):
        try:
            tts = gTTS(text=text, lang='en')
            audio_fp = "temp_audio.mp3"
            tts.save(audio_fp)  # Save the audio to a file
            os.system(f"start {audio_fp}")  # Use default system player to play audio
        except Exception as e:
            print(f"Error during text-to-speech playback: {e}")

class CameraFeed:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not access the webcam.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: Could not read frame from webcam.")
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

class Application:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.image_generator = ImageDescriptionGenerator(api_key)
        self.tts = TextToSpeech()
        self.camera = CameraFeed()
        print("Application initialized successfully.")

    def run(self):
        try:
            while True:
                frame = self.camera.get_frame()
                question = input("Ask a question (or type 'exit' to quit): ")
                if question.lower() == 'exit':
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                description = self.image_generator.generate_description(rgb_frame, prompt=question)
                print("Answer:", description)

                # Speak the description
                self.tts.speak(description)

                # Display frame with overlay text
                cv2.putText(frame, description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Camera Feed", frame)

        finally:
            self.camera.release()

# Entry point for the application
if __name__ == "__main__":
    app = Application()
    app.run()
