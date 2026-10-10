# 🎨 Neural Style Transfer Using Deep Learning

## 📌 Project Overview

Neural Style Transfer (NST) is a deep learning application that
combines the content of one image with the artistic style of another.

This project uses PyTorch and a pretrained VGG network to extract
visual features and generate stylized images.

The objective is to demonstrate how deep learning can be applied
to computer vision and artistic image generation.

## 🚀 Features

- Content image processing
- Style image processing
- Deep feature extraction using a pretrained VGG network
- Artistic style transfer
- Image preprocessing and normalization
- Stylized image generation
- PyTorch-based deep learning implementation

## 🧠 How It Works

1. Content and style images are loaded.
2. Images are preprocessed and normalized.
3. A pretrained VGG network extracts image features.
4. Content and style representations are combined using the
   project's style-transfer method.
5. The stylized image is generated and saved.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow (PIL)
- Deep Learning
- Convolutional Neural Networks (CNNs)
- Computer Vision

## 🏗️ Model Architecture

The project uses a pretrained VGG-based feature extractor to
capture content and style information from images.

The extracted features are used by the style-transfer pipeline
to produce an image that combines the content of one image
with the artistic characteristics of another.

## 📂 Project Structure

AI_NST_PROJECT/
├── utils/
├── experiments/
├── app.py
├── requirements.txt
└── README.md

Note: The actual file structure may vary depending on the
current version of the project.

## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/gaikwadsumitsudhakar-cpu/AI_NST_PROJECT.git

### 2. Navigate to the project directory

cd AI_NST_PROJECT

### 3. Create a virtual environment

python -m venv venv

### 4. Activate the environment on Windows

venv\Scripts\activate

### 5. Install dependencies

pip install -r requirements.txt

## ▶️ Run the Application

If app.py is the correct application entry point, run:

python app.py

Follow the application interface to provide the required images
and generate a stylized output.

## 🎯 Learning Outcomes

- Understanding CNN feature extraction
- Working with pretrained deep learning models
- Image preprocessing using PyTorch and Torchvision
- Applying deep learning to computer vision
- Understanding content and style representations
- Building and deploying a deep learning application

## 🔮 Future Improvements

- Support more artistic styles
- Improve inference speed
- Add batch image processing
- Compare multiple style-transfer approaches
- Improve the user interface

## 👨‍💻 Author

Sumit Gaikwad

MSc Physics – Final Year

Interested in Artificial Intelligence, Machine Learning,
Deep Learning, and Computer Vision.
