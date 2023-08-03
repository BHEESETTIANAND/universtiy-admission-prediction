import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

model = pickle.load(open("regressionL.pkl", "rb"))

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def main():
    st.title('Predicting masters admissions based on scores of various exams')
    st.write('Enter the required input to make a prediction:')
    
    feature1 = st.number_input("GRE score")
    feature2 = st.number_input("TOEFL score")
    feature3 = st.number_input("University rating")
    feature4 = st.number_input("SOP")
    feature5 = st.number_input("LOR")
    feature6 = st.number_input("CGPA")
    feature7 = st.number_input("Research")
    
    # Scale the data using the fitted StandardScaler
    scaled_data = scaler.transform([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]])
    
    if st.button('Make Prediction'):
        prediction = model.predict(scaled_data)[0]
        st.write(f"Your chance of getting into the university is {prediction}")

if __name__ == '__main__':
    main()
