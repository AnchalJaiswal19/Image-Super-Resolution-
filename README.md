# Image-Super-Resolution-
# 🖼️ Image Super Resolution using Color Intensity Enhancement

This project aims to enhance low-resolution images by increasing their color intensity, adjusting pixel values, and converting them into high-resolution images using Python, OpenCV, and deep learning techniques.

## 🚀 Project Objective

To develop a Super Resolution model that:
- Calculates the color intensity of images.
- Divides the image into smaller blocks using masking.
- Compares intensity between high- and low-resolution images.
- Enhances the low-resolution image based on intensity difference.

## 🧠 Technologies Used

- 🐍 Python
- 📸 OpenCV
- 📊 NumPy
- 📁 Google Colab (for development)
- 🧠 Convolutional Neural Network (CNN) (masking and enhancement logic)

## 🗂️ Dataset

The dataset includes image pairs of:
- Low Resolution: 720x720
- High Resolution: 1020x1020 and 840x840

> 💡 You can upload your own image pairs or use the provided dataset for training and testing.

## 📌 Features

- Block-wise masking of images
- Color intensity calculation
- Intensity difference computation
- Image enhancement using intensity values
- User-friendly interface via Colab notebook

## 📸 Sample Results

| Low Resolution Image | Enhanced Image |
|----------------------|----------------|
| ![Low Res](low_res.jpg) | ![Enhanced](enhanced.jpg) |


## 🏗️ Future Improvements
- Integration of a pre-trained SRGAN or ESRGAN model.
- Web interface to upload and enhance images directly.
- Real-time video super resolution support.

## Author
- A passionate tech enthusiast focusing on computer vision, image processing, and AI-based web development.
📧 Email-anchaljaiswal4040@gmail.com
