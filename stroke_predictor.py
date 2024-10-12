import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('./XGBoost.pkl')

# Define options for user input
sex_options = {1: 'Male', 2: 'Female'}
race_options = {1: 'Hispanic', 2: 'White only, non-Hispanic', 3: 'Black only, non-Hispanic', 4: 'Other Multiracial'}
education_options = {1: 'Did not graduate High School', 2: 'Graduate High School'}
income_options = {1: '<$25000', 2: '$25000≤income<$50000', 3: '≥$50000'}
marital_options = {1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never married', 6: 'Living with partner'}
exercise_options = {1: 'Yes', 2: 'No'}
alcohol_options = {1: 'Yes', 2: 'No'}
diabetes_options = {1: 'Yes', 2: 'No'}
heart_disease_options = {1: 'Yes', 2: 'No'}
smoking_options = {1: 'Non-smoker', 2: 'Only combustible cigarettes', 3: 'Only e-cigarettes', 4: 'Both combustible cigarettes and e-cigarettes'}

# User interface
st.title("Stroke Risk Prediction")

# Input features
age65yr = st.selectbox("Age (65 years or older)", options=[1, 2], format_func=lambda x: "18≤age<65" if x == 1 else "≥65")
sex = st.selectbox("Sex", options=list(sex_options.keys()), format_func=lambda x: sex_options[x])
race = st.selectbox("Race", options=list(race_options.keys()), format_func=lambda x: race_options[x])
educat = st.selectbox("Educational Level", options=list(education_options.keys()), format_func=lambda x: education_options[x])
income = st.selectbox("Income", options=list(income_options.keys()), format_func=lambda x: income_options[x])
marital = st.selectbox("Marital Status", options=list(marital_options.keys()), format_func=lambda x: marital_options[x])
exercise = st.selectbox("Exercise", options=list(exercise_options.keys()), format_func=lambda x: exercise_options[x])
weight = st.number_input("Weight (kg)", min_value=0.0, format="%.2f")
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
alcohol = st.selectbox("Alcohol Consumption", options=list(alcohol_options.keys()), format_func=lambda x: alcohol_options[x])
stroke = st.selectbox("History of Stroke", options=[1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
diabetes = st.selectbox("Diabetes", options=list(diabetes_options.keys()), format_func=lambda x: diabetes_options[x])
total_heart = st.selectbox("Total Heart Disease", options=list(heart_disease_options.keys()), format_func=lambda x: heart_disease_options[x])
smoking_type = st.selectbox("Type of Smoking", options=list(smoking_options.keys()), format_func=lambda x: smoking_options[x])

if st.button("Predict Stroke Risk"):
    # Construct input data
    input_data = pd.DataFrame({
        'AGE65YR': [age65yr],
        'SEX': [sex],
        'RACE': [race],
        'EDUCAT': [educat],
        'INCOME': [income],
        'MARITAL': [marital],
        'EXERCISE': [exercise],
        'WEIGHT': [weight],
        'BMI': [bmi],
        'Alcohol': [alcohol],
        'STROKE': [stroke],
        'DIABETES': [diabetes],
        'Total Heart': [total_heart],
        '吸烟类型': [smoking_type]
    })

    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = prediction[0]  # Assuming the model returns a single prediction

    # Calculate the probability (if your model can return probabilities)
    probability = model.predict_proba(input_data)[:, 1][0] * 100  # Adjust index as needed

    # Display prediction result
    if predicted_class == 1:
        advice = (            
            f"According to our model, you have a high risk of STROKE. "            
            f"The model predicts that your probability of having STROKE is {probability:.1f}%. "            
            "While this is just an estimate, it suggests that you may be at significant risk. "            
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "           
            "to ensure you receive an accurate diagnosis and necessary treatment."        
        )
    else:
        advice = (            
            f"According to our model, you have a low risk of stroke. "
            f"The model predicts that your probability of not having stroke is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "           
            "I recommend regular check-ups to monitor your health, "          
            "and to seek medical advice promptly if you experience any symptoms." 
        )
    st.write(advice)
