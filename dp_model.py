import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import Tk, Button, Label, filedialog, messagebox
from PIL import Image

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10

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
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir):
    train_data = load_data(data_dir)
    model = create_model()
    model.fit(train_data, epochs=EPOCHS)
    model.save('mri_model.h5')
    return model

def predict_image(model, img_path):
    img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, IMG_HEIGHT, IMG_WIDTH, 3))
    prediction = model.predict(img_array)
    return 'Yes' if prediction[0][0] > 0.5 else 'No'

def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        model = keras.models.load_model('mri_model.h5')
        result = predict_image(model, file_path)
        messagebox.showinfo("Prediction Result", f"The model predicts: {result}")

root = Tk()

root.title("MRI Image Classifier")
root.geometry("300x200")

train_button = Button(root, text="Train Model", command=lambda: train_model('dataset'))
train_button.pack(pady=10)

predict_button = Button(root, text="Upload MRI Image", command=upload_and_predict)
predict_button.pack(pady=10)

root.mainloop()
