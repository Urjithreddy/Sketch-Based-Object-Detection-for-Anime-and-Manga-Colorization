import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "generator.onnx"
ort_session = ort.InferenceSession(MODEL_PATH)

def do_inference(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    image = np.array(image).astype(np.float32)
    image = image / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)
    
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_out = ort_session.run(None, ort_inputs)
    
    out = ort_out[0]
    out = (out + 1.0) / 2.0
    out = np.clip(out[0] * 255, 0, 255).astype(np.uint8)
    
    return out

st.title("Sketch Based Object Detection for Anime & Manga Colorization")
st.write("Upload a black-and-white image to get the colorized version.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Processing..."):
        output_img = do_inference(uploaded_file)
    
    st.image(output_img, caption="Colorized Image", use_column_width=True)
