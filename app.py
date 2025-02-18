import streamlit as st
import onnxruntime
from torchvision import transforms
from model_inference import lab_to_rgb, get_input_tensor, to_numpy
import torch
from PIL import Image
import numpy as np
import os  # Import the os module

# Load the ONNX model
model_path = "d:/pragnya/capstone project/Manga-Colorizer-main/models/generator.onnx"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No model file found at '{model_path}'")

ort_session = onnxruntime.InferenceSession(model_path)

st.title("Enhanced Anime-Manga Colorization Using Pix2Pix")
st.write(
    ""
)
st.write(
    ""
)
st.markdown(
    ""
)
file_up = st.file_uploader("Upload an image", type=["jpg", "png"])


def do_inference(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Keep the image in RGB format
    image = image.resize((1024, 1024))  # Resize to 1024x1024 to match the model's expected input size
    image = np.array(image)
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]

    # Print the shape of the input image tensor
    print(f"Input image tensor shape: {image_tensor.shape}")

    # Perform inference
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image_tensor)}
    ort_output = ort_session.run(None, ort_inputs)
    fake_ab = ort_output[0]
    fake_ab = torch.from_numpy(fake_ab)

    # Print the shape of the fake_ab tensor
    print(f"Fake AB tensor shape: {fake_ab.shape}")

    L = image_tensor[:, :1, :, :]  # Extract the L channel from the input image

    # Print the shape of the L tensor
    print(f"L tensor shape: {L.shape}")

    # Ensure fake_ab has 2 channels
    fake_ab = fake_ab[:, :2, :, :]

    # Print the shape of the adjusted fake_ab tensor
    print(f"Adjusted Fake AB tensor shape: {fake_ab.shape}")

    rgb_img = lab_to_rgb(L, fake_ab)  # Ensure correct input to lab_to_rgb

    # Print the shape of the RGB image tensor
    print(f"RGB image tensor shape: {rgb_img.shape}")

    img = transforms.ToPILImage()(rgb_img.squeeze(0))
    org_img = transforms.ToPILImage()(image_tensor.squeeze(0))
    st.image(
        [org_img, img],
        caption=["Original Image", "Generated Image"],
        use_column_width=False,
    )


if file_up is None:
    st.write("")
else:
    do_inference(file_up)