import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
import pyttsx3
import tempfile
import os
import cv2
import numpy as np

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Convert to numpy array for OpenCV processing
    image_np = np.array(image)
    # Deskew the image
    coords = np.column_stack(np.where(image_np > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_np = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Apply adaptive thresholding
    image_np = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Apply Otsu's binarization
    _, image_np = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    image_np = cv2.dilate(image_np, kernel, iterations=1)
    image_np = cv2.erode(image_np, kernel, iterations=1)
    # Convert back to PIL image
    image = Image.fromarray(image_np)
    # Resize image
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    return image

def extract_text_from_image(image):
    image = preprocess_image(image)
    text = pytesseract.image_to_string(image, lang='eng')
    return text

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image = preprocess_image(image)
        text += pytesseract.image_to_string(image, lang='eng')
    return text

def text_to_speech(text, output_audio_path):
    try:
        if not text.strip():
            raise ValueError("The extracted text is empty.")
        
        engine = pyttsx3.init()
        engine.save_to_file(text, output_audio_path)
        engine.runAndWait()
        
        if not os.path.exists(output_audio_path):
            raise FileNotFoundError(f"Audio file was not created at {output_audio_path}")
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")

def main():
    st.title("📄 Document Recognition and Text-to-Speech App")
    st.markdown("### Upload a PDF or Image file to extract text and convert it to speech 📄🔊")

    uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(temp_file_path)
        else:
            image = Image.open(temp_file_path)
            text = extract_text_from_image(image)

        st.text_area("📝 Extracted Text", text, height=200)

        if st.button("Convert to Speech 🔊"):
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'female' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate * 0.6)  # Decrease speed by 40%
            
            engine.say(text)
            engine.runAndWait()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                text_to_speech(text, temp_audio_file_path)
                if os.path.exists(temp_audio_file_path):
                    st.audio(temp_audio_file_path, format='audio/mp3')
                else:
                    st.error("Audio file was not created.")

if __name__ == "__main__":
    main()