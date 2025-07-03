import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
import openai
import base64
import io
import os

# Set your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set it directly here (not recommended)

# Optional: Set tesseract path manually if needed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="AI Blueprint Assistant", layout="centered")
st.title("🏗️ AI Blueprint Assistant")
st.markdown("Upload a site plan or blueprint to extract data and ask questions.")

uploaded_file = st.file_uploader("Upload a blueprint (PNG, JPG, PDF)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, caption="Uploaded Blueprint", use_column_width=True)

    if len(img_array.shape) == 2:
    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
elif img_array.shape[2] == 4:
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # OCR text extraction
    ocr_text = pytesseract.image_to_string(image)
    st.subheader("📄 Detected Text:")
    st.text(ocr_text[:1000])

    # Rectangle detection (simple contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rectangles += 1

    st.subheader(f"📏 Detected Zones: {rectangles} possible rectangles")
    st.image(img_array, caption="Annotated Blueprint", use_column_width=True)

    # Chat with GPT-4 Vision
    with st.expander("💬 Ask AI about this blueprint"):
        question = st.text_input("Enter your question about the blueprint")

        if question:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            img_url = f"data:image/png;base64,{img_b64}"

            from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {"role": "system", "content": "You are an expert..."},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": img_url}}
        ]}
    ],
    max_tokens=500
)

            answer = response.choices[0].message.content
            st.markdown("**AI Response:**")
            st.write(answer)
