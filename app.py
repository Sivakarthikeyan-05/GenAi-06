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
    try:
        prompt = "Write a short, descriptive caption for this image."
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt]
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

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
