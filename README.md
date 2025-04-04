# Sketch-Based-Object-Detection-for-Anime-and-Manga-Colorization

This repository provides a complete pipeline for colorizing black and white images using deep learning. It features multiple model architectures, robust data preprocessing, and a deployment interface to showcase the results. Whether you're a researcher or an enthusiast, you'll find the code modular and easy to extend.

## Overview

The project implements three distinct approaches to image colorization:

- **Pix2Pix GAN**:  
  An adversarial framework using a U-Net based generator and a PatchGAN discriminator. The generator learns to produce realistic color images, while the discriminator evaluates the authenticity of the generated images.

- **Patch-Based Model**:  
  This model divides images into smaller 64x64 patches. By learning from these patches, the model can capture local color patterns and details more effectively.

- **Custom Fused Model**:  
  A unique architecture that combines multiple convolutional streams. It fuses deep features from two branches before reconstructing the color image, leveraging both global and local information for improved results.

## Key Features

- **Data Preprocessing**:  
  Automatically pairs black-and-white images with their corresponding colored versions from a specified dataset folder. The images are resized, normalized, and prepared for training.

- **Model Architectures**:  
  Detailed implementations for building the generator, discriminator, patch-based, and custom fused models are provided. Each model is designed with simplicity and performance in mind.

- **Training and Evaluation**:  
  Custom training loops are implemented for each model. The pipeline includes:
  - Training functions that run over multiple epochs.
  - A loss function combining adversarial loss and L1 loss for the Pix2Pix model.
  - Functions to compute the Structural Similarity Index (SSIM) as an evaluation metric.

- **Model Conversion and Deployment**:  
  The custom fused model is converted from a TensorFlow/Keras format to ONNX, making it compatible with a wide range of deployment environments. Additionally, a simple Streamlit application is included to allow users to upload an image and receive a colorized output in real time.

## Repository Structure

- **Data Processing**:
  - `load_image_pairs()`: Scans the dataset directory to find matching pairs of black-and-white and colored images.
  - `preprocess_image()`: Loads, resizes, and normalizes images for model consumption.
  - `preprocess_dataset()`: Aggregates preprocessed image pairs into NumPy arrays.

- **Model Building**:
  - `build_generator()`: Constructs the U-Net-based generator model for the Pix2Pix framework.
  - `build_discriminator()`: Builds the discriminator using a PatchGAN approach.
  - `build_patch_based_model()`: Creates a model that processes image patches.
  - `build_custom_model()`: Develops a fused architecture combining multiple convolutional layers.

- **Training Routines**:
  - `train_pix2pix()`: Manages the training loop for the Pix2Pix model with adversarial and L1 losses.
  - `train_patch_based_model()`: Trains the patch-based model on image patches.
  - `train_custom_model()`: Handles the training for the custom fused model.

- **Evaluation and Testing**:
  - `compute_ssim()`: Computes the SSIM score to assess the similarity between the generated and target images.
  - `test_models()`: Runs comprehensive tests on the three models, displays results, and prints evaluation metrics.

- **Model Conversion**:
  - A dedicated script converts the custom fused model to ONNX format using `tf2onnx`, facilitating integration into different deployment scenarios.

- **Deployment with Streamlit**:
  - A lightweight web app built with Streamlit allows users to interact with the ONNX model. Users can upload an image and view the colorized output immediately.

## Evaluation Metrics

After training, the models achieved the following SSIM accuracies:

- **Pix2Pix Generator**: 85.07%
- **Patch-Based Model**: 82.38%
- **Our Model**: 95.01%

These metrics demonstrate the superior performance of the Custom Fused Model in terms of structural similarity to the ground truth images.

## Sample Output

Below is an area where you can paste the output image showcasing the colorization result:

![Colorized Output](https://github.com/user-attachments/assets/67364c8b-febf-46ee-b57a-44d111ce34e2)



## How to Use

1. **Dataset Preparation**:  
   Organize your dataset in a directory (e.g., `smalldatasetmanga`) where each pair of images is named following the pattern:  
   - Black-and-white images: `bw_image_#.png`
   - Color images: `color_image_#.png`

2. **Training the Models**:  
   Run the main script to:
   - Load and preprocess the dataset.
   - Train each model for a specified number of epochs.
   - Evaluate the performance using SSIM.
   - Save the trained models in H5 format.

3. **Converting to ONNX**:  
   Use the provided script to convert the custom fused model to ONNX format. This file can then be deployed using various inference frameworks.

4. **Deploying with Streamlit**:  
   Launch the Streamlit app by running `streamlit run <app_script>.py`. The app allows you to upload a black-and-white image and see the colorized output generated by the ONNX model.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- PIL (Pillow)
- Matplotlib
- tf2onnx
- Streamlit
- ONNX Runtime

## Additional Notes

- **Extensibility**:  
  The modular nature of the code makes it easy to experiment with new architectures or training techniques. Researchers can modify the existing models or add new modules.

- **Evaluation Metrics**:  
  The inclusion of SSIM ensures that you can quantitatively assess the quality of the colorized images compared to the ground truth.

- **Deployment Flexibility**:  
  By converting the model to ONNX and providing a Streamlit interface, the repository bridges the gap between research and real-world application, allowing easy integration and demonstration.

## Conclusion

This project demonstrates a robust approach to image colorization using deep learning. With multiple models, comprehensive training routines, and a user-friendly web interface, it's a valuable resource for anyone interested in applying neural networks to creative tasks. Explore, experiment, and feel free to enhance the models further!

