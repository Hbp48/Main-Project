# import os
# from dotenv import load_dotenv
# from PIL import Image
# import streamlit as st
# import pyttsx3
# import google.generativeai as genai
# from concurrent.futures import ThreadPoolExecutor

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# model = genai.GenerativeModel("gemini-pro-vision")

# # Create a thread pool executor to manage the text-to-speech engine
# engine = pyttsx3.init()
# engine.setProperty("rate", 200)
# engine.setProperty("volume", 1.0)
# executor = ThreadPoolExecutor(max_workers=1)

# def get_gemini_response(input_text, image):
#     if input_text != "":
#         response = model.generate_content([input_text, image])
#     else:
#         response = model.generate_content(image)
    
#     if response == None:
#         return ""

#     return response.text

# def speak(text):
#     def _speak():
#         engine.say(text)
#         engine.runAndWait()
#     executor.submit(_speak)

# # Main Streamlit application code
# st.set_page_config(page_title="Q&A LLM")
# st.header("Welcome to Accesify application")

# # Speak the welcome message
# speak("Welcome to Accesify application")

# upload_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if upload_file is not None:
#     image = Image.open(upload_file)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)

#     speak("Captioning the image...")
#     response = get_gemini_response("Do the image captioning", image)
#     st.subheader("The Image captioning is")
#     st.write(response)
#     speak(response)

#     speak("Doing OCR on the image...")
#     response = get_gemini_response("Do the ocr", image)
#     st.subheader("OCR")
#     st.write(response)
#     speak(response)

#     speak("If you want to provide a prompt, enter it now.")

#     input_text = st.text_input("Input: ", key="input")

#     speak("Press the submit button to process the image and prompt.")
#     submit = st.button("Submit")

#     if submit:
#         response = get_gemini_response(input_text, image)
#         st.subheader("The Response is")
#         st.write(response)
    #      speak(response)

print("you")
