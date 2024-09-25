# predict.py
import joblib
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import os

def preprocess_image(image_path):
    img = imread(image_path)
    img_resized = resize(img, (64, 64, 3), preserve_range=True)
    img_normalized = img_resized / 255.0
    return img_normalized.flatten()

def load_model():
    model = joblib.load('svm_cats_dogs_model.pkl')
    print("Model loaded from 'svm_cats_dogs_model.pkl'")
    return model

def predict_image(model, image_path):
    image_data = np.array([preprocess_image(image_path)])
    prediction = model.predict(image_data)
    return "Cat" if prediction == 0 else "Dog"

if __name__ == "__main__":
    model = load_model()
    image_path = '4.jpg'
    prediction = predict_image(model, image_path)
    print(f"Prediction: {prediction}")
