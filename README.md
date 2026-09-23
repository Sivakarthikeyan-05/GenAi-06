# Prototype Development for Image Captioning Using the BLIP Model and Gradio Framework

## AIM:
To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

## PROBLEM STATEMENT:
Developing an accessible and interactive web application that can generate descriptive text (captions) for uploaded images to aid in image understanding and accessibility. 

## DESIGN STEPS:
**STEP 1:** Install necessary libraries including `gradio`, `google-genai`, `python-dotenv`, and `Pillow`.
**STEP 2:** Authenticate and initialize the Gemini API client with the provided API key for generating captions.
**STEP 3:** Define a function to take an image as input, send it to the model with a prompt to generate a caption, and return the generated text.
**STEP 4:** Create a Gradio interface (`gr.Interface`) to upload images, link it to the captioning function, and launch the web server.

## PROGRAM:
```python
import os
import gradio as gr 
from PIL import Image
from google import genai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

# Configure the API key from environment variable
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API Key not found. Please set the GEMINI_API_KEY environment variable in your .env file.")
client = genai.Client(api_key=api_key)

def captioner(image):
    prompt = "Write a short, descriptive caption for this image."
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[image, prompt]
    )
    return response.text

gr.close_all()
demo = gr.Interface(fn=captioner,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with Gemini",
                    description="Caption any image using the Gemini model",
                    flagging_mode="never",
                    examples=["robo.jpeg", "strawberry.jpg", "calculate.png"])

if __name__ == "__main__":
    port = int(os.environ.get('PORT1', 7860))
    demo.launch(share=True, server_port=port)
```

## OUTPUT:

<img width="1090" height="553" alt="image" src="https://github.com/user-attachments/assets/723c9b0b-0e59-4ab8-806d-be3cfd24846f" />


## RESULT:
The prototype application for image captioning was successfully designed and deployed using the Gradio UI framework and the Gemini API.
