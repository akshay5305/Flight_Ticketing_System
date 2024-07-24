import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
import io


# Loading the saved model
try:
    loaded_model = pickle.load(open('flight_price_prediction_model.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Creating a function for prediction
def flight_price_prediction(input_data):
    try:
        # Changing the input data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # Reshape the array as we are predicting on one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Predicting the flight ticket price
        prediction = loaded_model.predict(input_data_reshaped)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def load_image(image_path):
    try:
        with open(image_path, "rb") as file:
            image_data = file.read()
            if not image_data:
                st.error(f"Image file {image_path} is empty.")
                return None
            return Image.open(io.BytesIO(image_data))
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

def main():

    # Loading and displaying the background image
    bg_image = load_image('bg.png')
    if bg_image:
        st.image(bg_image)

    # Giving a title
    st.title('Flight Ticket Price Prediction')

    # Getting input data from user
    Total_Stops = st.number_input("Total Stops")
    Date = st.number_input("Date")
    Month = st.number_input("Month")
    Year = st.number_input("Year")
    Dep_hours = st.number_input("Departure Hours")
    Dep_min = st.number_input("Departure Minutes")
    Arrival_hours = st.number_input("Arrival Hours")
    Arrival_min = st.number_input("Arrival Minutes")
    Duration_hours = st.number_input("Duration Hours")
    Duration_min = st.number_input("Duration Minutes")

    # Code for prediction
    price = ''

    # Creating a button for Prediction
    if st.button('Predict Flight Price'):
        price = flight_price_prediction([Total_Stops, Date, Month, Year, Dep_hours, Dep_min, Arrival_hours, Arrival_min, Duration_hours, Duration_min])
        if price is not None:
            st.success(f'The Predicted Flight Ticket Price: {price}$')

    # Displaying images
    st.subheader('Model Statistics:')
    for img_path, caption in [('111.png', ' '), ('112.png', ' ')]:
        img = load_image(img_path)
        if img:
            st.image(img, caption=caption)

if __name__ == '__main__':
    main()
