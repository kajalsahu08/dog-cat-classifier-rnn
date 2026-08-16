import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('cat_dog_rnn.h5')

# App title
st.title("🐶 Dog vs Cat Classifier 🐱")
st.write("Upload an image and the model will predict if it's a Dog or a Cat!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    # Preprocess the image
    img_size = 128
    image = image.convert('L')  # convert to grayscale
    image = image.resize((img_size, img_size))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, img_size, img_size)

    # Make prediction
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("Prediction: 🐶 It's a Dog!")
    else:
        st.success("Prediction: 🐱 It's a Cat!")
