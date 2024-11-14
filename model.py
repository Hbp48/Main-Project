import cv2
import os
from PIL import Image
import google.generativeai as genai
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API key for the Generative AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel('gemini-1.5-flash-8b')

def generate_description_for_frame(frame, prompt="Explain this image within 30 words"):
    """Generate a description for a single image frame using the gemini-1.5-flash model."""
    pil_image = Image.fromarray(frame)
    response = model.generate_content([prompt, pil_image])  
    return response.text

def save_description_as_audio(description, filename="description_audio.mp3"):
    """Convert description text to speech and save it as an audio file."""
    tts = gTTS(text=description, lang='en')
    tts.save(filename)
    print(f"Audio saved as {filename}")
    return filename

# Open a named window and access the webcam
cv2.namedWindow("Live Feed")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    print("Error: Could not access the webcam.")
    rval = False

try:
    while rval:
        cv2.imshow("Live Feed", frame)
        rval, frame = vc.read()

        # Check for user input to capture frame or exit
        key = cv2.waitKey(20)
        
        # Press 'c' to capture frame and ask a question
        if key == ord('c'):
            captured_frame = frame.copy()
            question = input("Ask a question about the captured frame: ")
            
            # Prepare the prompt with the question
            prompt = f"Answer the question: {question}. Limit to 30-35 words."
            print(f"Question being asked: {question}")

            # Generate description for the captured frame
            description = generate_description_for_frame(captured_frame, prompt=prompt)
            print("Answer:", description)

            # Save the description as audio
            save_description_as_audio(description)

        # Press 'ESC' to exit
        if key == 27:  # ESC key
            print("Exiting the live feed.")
            break

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    # Release resources and close the video window
    vc.release()
    cv2.destroyWindow("Live Feed")
