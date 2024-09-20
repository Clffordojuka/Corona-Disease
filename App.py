import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib

st.title("Corona Virus Disease Prediction")

# Load models
rf_model = joblib.load('random_forest_model.pkl')
ridge_model = joblib.load('ridge_model.pkl')
lasso_model = joblib.load('lasso_model.pkl')
elastic_model = joblib.load('elasticnet_model.pkl')

# Sidebar to select the model
model_choice = st.sidebar.selectbox('Choose a model', 
                                    ['Random Forest', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression'])

# User inputs
st.sidebar.header("Input Parameters")
def user_input_features():
    Confirmed = st.sidebar.number_input('Confirmed Cases', min_value=0, step=1)
    Recovered = st.sidebar.number_input('Recovered Cases', min_value=0, step=1)
    Active = st.sidebar.number_input('Active Cases', min_value=0, step=1)
    recovery_rate = st.sidebar.number_input('recovery_rate', min_value=0, step=1)
    active_case_rate = st.sidebar.number_input('active_case_rate', min_value=0, step=1)
    log_confirmed = st.sidebar.number_input('log_confirmed', min_value=0, step=1)

    data = {'Confirmed': Confirmed,
            'Recovered': Recovered,
            'Active': Active,
            'recovery_rate': recovery_rate,
            'active_case_rate': active_case_rate,
            'log_confirmed': log_confirmed}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input:')
st.write(input_df)

# Make predictions based on the model choice
if model_choice == 'Random Forest':
    prediction = rf_model.predict(input_df)
elif model_choice == 'Ridge Regression':
    prediction = ridge_model.predict(input_df)
elif model_choice == 'Lasso Regression':
    prediction = lasso_model.predict(input_df)
elif model_choice == 'ElasticNet Regression':
    prediction = elastic_model.predict(input_df)

# Display the prediction
st.subheader('Prediction')
st.write(f"Predicted Deaths: {prediction[0]:.2f}")
