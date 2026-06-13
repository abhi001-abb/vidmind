from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai

print("Starting...")

api_key = os.getenv("GOOGLE_API_KEY")
print("API key found:", bool(api_key))

genai.configure(api_key=api_key)

print("Creating model...")
model = genai.GenerativeModel("gemini-2.5-flash")

print("Sending request...")
response = model.generate_content("Say hello")

print("Received response")
print(response.text)
