# Image Denoising using Convolutional Autoencoder

## Project Overview

This project implements a Convolutional Autoencoder to remove Gaussian noise from handwritten digit images using the MNIST dataset.

The model learns to reconstruct clean images from noisy inputs using an encoder-decoder architecture.

---

## Dataset

- MNIST Dataset
- 60,000 training images
- 10,000 testing images
- Image Size: 28×28 pixels

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab

---

## Model Architecture

Encoder:
- Conv2D (32 filters)
- MaxPooling2D
- Conv2D (16 filters)
- MaxPooling2D

Decoder:
- Conv2D (16 filters)
- UpSampling2D
- Conv2D (32 filters)
- UpSampling2D
- Conv2D (1 filter, Sigmoid)

---

## Training Configuration

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 20
- Batch Size: 128

---

## Results

The model successfully removes Gaussian noise from MNIST images while preserving the digit structure.

Training Loss: ~0.099

Validation Loss: ~0.098

---

## Project Workflow

1. Load MNIST dataset
2. Normalize images
3. Add Gaussian noise
4. Build Encoder
5. Build Decoder
6. Train Autoencoder
7. Generate denoised images
8. Visualize results

---

## Author

Riya Agrawal
