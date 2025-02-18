import segmentation_models_pytorch as smp
import torch
from datasets import load_dataset
import onnx
import os
import numpy as np
from PIL import Image

def convert_to_onnx(model_state_dict, inputs):
    """Method to convert pytorch models to onnx format.
    Args:
        model_state_dict: The state dictionary of model.
        inputs: The input tensor
    """
    model = smp.Unet(
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        in_channels=3,  # Ensure the model expects 3-channel RGB input
        classes=3,  # Update to 3 channels for RGB output
        decoder_attention_type="scse"  # Add attention to the decoder
    )

    model.load_state_dict(model_state_dict)
    model.eval()
    torch.onnx.export(
        model,
        inputs,
        "d:/pragnya/capstone project/Manga-Colorizer-main/models/generator.onnx",
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

if __name__ == "__main__":
    # Load the generator model state dictionary
    model_path = "d:/pragnya/capstone project/Manga-Colorizer-main/models/generator_model.bin"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at '{model_path}'")
    
    print(f"Loading model from {model_path}")
    
    ckpt = torch.load(model_path)
    print(f"Checkpoint keys: {ckpt.keys()}")
    
    # Load the dataset from Hugging Face
    dataset_name = "MichaelP84/manga-colorization-dataset"
    dataset = load_dataset(dataset_name, split="train[:1%]")  # Load a small subset for testing
    
    # Print the keys of the first sample to understand its structure
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    
    # Use the 'bw_image' key to get the input image
    image = sample["bw_image"]
    image = Image.fromarray(np.array(image)).convert("RGB")  # Convert to RGB
    image = np.array(image)
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]

    convert_to_onnx(model_state_dict=ckpt, inputs=image_tensor)  # Pass the entire checkpoint dictionary
    
    # Check if onnx works
    onnx_model = onnx.load("d:/pragnya/capstone project/Manga-Colorizer-main/models/generator.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model exported and checked successfully.")