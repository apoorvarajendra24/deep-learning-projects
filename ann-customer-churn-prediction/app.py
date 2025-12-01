"""
Q. What is Streamlit?
A Python library to build web apps quickly
Used for ML model deployment, dashboards, forms
No HTML/CSS required 
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st

model = load_model('model.h5')
with open('Label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('One_hot_encoder_geography.pkl','rb') as file:
    one_hot_encoder_geography=pickle.load(file)
with open('Scaler.pkl','rb') as file:
    scaler=pickle.load(file)


# Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number Of Products',1,4)
has_credit_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# Input data frame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = one_hot_encoder_geography.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
predict_proba = prediction[0][0]
st.write(f'CUstomer Churm Probablility: {predict_proba:.2f}')
if predict_proba > 0.5:
    st.write("The Customer is likely to Churn.")
else:
    st.write("The Customer is not likely to churn.")