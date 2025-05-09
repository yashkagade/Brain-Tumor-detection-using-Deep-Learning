# 🧠 Brain-Tumor-Detection-Using-Deep-Learning

## 📌 Design Project Submission  
This project is developed as a part of the **Design Project** for the academic year 2024–25 by:

- **Yash Kagade**  
- **Atharva Jathar**  
- **Harshad Lolage**  
- **Sahil Paradhi**

Under the guidance of **Prof. Prranjali Jadhav (Ma'am)**  
Department of (AI & DS),  
Vishwakarma Institute of Information Technology, Pune.

---

## 📂 Project Overview
This is a Deep Learning-based web application that detects the presence of brain tumors in MRI scans. The model is trained using Convolutional Neural Networks (CNNs) and integrated with a user-friendly interface built using **Streamlit**. The application allows users to:

- Train a model on MRI image datasets.
- Upload an MRI image.
- Get an instant prediction on whether the tumor is present or not.

---

## 🧠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pillow (PIL)**
- **Streamlit**

---

## 🏗️ Model Architecture

The CNN model consists of:

- Input Layer  
- 2 Convolutional Layers (with ReLU activation and MaxPooling)  
- Flatten Layer  
- Dense Layer (128 neurons, ReLU)  
- Output Layer (Sigmoid activation for binary classification)

---

## 🧪 How It Works

1. **Dataset Loading** – The dataset is loaded from a specified folder using `ImageDataGenerator`.
2. **Model Training** – The CNN model is trained for 30 epochs and saved as `mri_model.h5`.
3. **Prediction** – Uploaded MRI images are preprocessed and passed through the model for prediction.
4. **Interface** – Streamlit is used to provide a simple UI for training the model and uploading images for classification.

---

## Directory Structure

Brain-Tumor-Detection/
├── dataset/             # Folder containing MRI image dataset
├── mri_model.h5         # Trained model (auto-generated)
├── app.py               # Main Streamlit application
└── README.md            # Project documentation


