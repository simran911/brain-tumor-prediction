import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('BrainTumor10Epochs.h5')

def main():
    st.title("Simran's Brain Tumor Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        resized_image = image.resize((100, 100))
        st.image(resized_image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img = image.resize((64, 64))
        img_array = np.array(img)
        input_img = np.expand_dims(img_array, axis=0)

        # Make prediction
        result = model.predict(input_img)

        # Display the prediction result
        if result[0][0] == 1:
            st.success("Prediction: Tumor detected")
        else:
            st.success("Prediction: No tumor detected")

if __name__ == "__main__":
    main()
