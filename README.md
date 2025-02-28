Haze Detection and Removal with CNN & DCP

This project combines deep learning and classical image processing to detect and remove haze from images. A Convolutional Neural Network (CNN) is used for haze classification, and the Dark Channel Prior (DCP) algorithm enhances image clarity by removing detected haze.

📌 Project Summary

Hazy images can degrade visibility and impact computer vision tasks. This project offers a dual approach:

Haze Detection: A CNN model classifies images as hazy or clear.

Haze Removal: If haze is detected, the DCP algorithm restores image clarity.

🛠 Requirements

Ensure you have the following dependencies installed:

Python

PyTorch

Pillow

scikit-image

NumPy

SciPy

Matplotlib

Install all dependencies using:

pip install -r requirements.txt

🚀 Installation & Setup

Clone the repository:

git clone https://github.com/ilknurpehlivan/Haze-Detection-and-Removal-with-CNN-and-DCP.git

📂 Datasets

The project utilizes publicly available datasets:

Dense Haze Dataset (NTIRE 2019 Challenge)

Provides real-world and synthetic hazy images.

Dataset Link

RESIDE Standard Dataset

Includes both real and synthetic hazy images with ground truth.

Dataset Link

🔹 Organizing Data

Download and extract the datasets, then structure them in the data folder as follows:

data/
 ├── hazy_images/
 ├── clear_images/

🏋 Training the Model

Train the CNN model using:

python train.py

🖥 How It Works

1️⃣ Haze Detection

The CNN model predicts whether an image is hazy or clear.

2️⃣ Haze Removal

If an image is hazy, the DCP algorithm estimates the transmission map and atmospheric light to restore visibility.

📸 Sample Interface

Here’s how the application interface looks:



