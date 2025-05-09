import os
import numpy as np
import streamlit as st
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import time

# Constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = "mri_model.h5"
DATASET_PATH = "dataset"  # Change to actual dataset path

def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    return train_data

def create_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_layer"),
        layers.Conv2D(32, (3, 3), activation='relu', name="conv2d_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
        layers.Conv2D(64, (3, 3), activation='relu', name="conv2d_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation='relu', name="dense_1"),
        layers.Dense(1, activation='sigmoid', name="output")
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    if not os.path.exists(DATASET_PATH):
        st.error("Dataset directory not found!")
        return None
    
    st.write("Loading dataset...")
    train_data = load_data(DATASET_PATH)
    
    st.write("Training model...")
    progress_bar = st.progress(0)
    model = create_model()
    
    for i in range(EPOCHS):
        model.fit(train_data, epochs=1, verbose=0)
        progress_bar.progress((i + 1) / EPOCHS)
        time.sleep(0.5)  # Simulating training time delay
    
    model.save(MODEL_PATH)
    st.success("Model trained and saved!")
    return model

def predict_image(model, img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, IMG_HEIGHT, IMG_WIDTH, 3))
    prediction = model.predict(img_array)
    return 'Yes' if prediction[0][0] > 0.5 else 'No'

st.title("🧠 MRI Brain Tumor Detection")

# Train Model Button
if st.button("Train Model"):
    model = train_model()

# Load model if exists
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:
    model = None

# Upload and Predict
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
if uploaded_file and model:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    
    # Create a placeholder for the "Classifying..." text
    status_text = st.empty()

    with st.spinner("Classifying..."):
        prediction = predict_image(model, image)
    
    # Remove "Classifying..." and show result
    status_text.empty()
    st.success(f"The model predicts: **{prediction}**")
