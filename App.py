import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import joblib # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Title of the app
st.title("Corona Virus Disease Prediction")

# Sidebar Navigation
st.sidebar.title("Navigation") 
navigation = st.sidebar.radio("Go to", ["Homepage", "Data Information", "Visualization", "Machine Learning Model"])

# Homepage section
if navigation == "Homepage":
    st.header("Welcome to the COVID-19 Prediction App")
    st.write("""
    This app predicts the number of deaths due to COVID-19 based on the provided data.
    You can navigate to different sections using the sidebar to explore the data, visualize insights, and interact with the machine learning models.
    """)

# Data Information section
elif navigation == "Data Information":
    st.header("Data Information")
    st.write("""
    **Confirmed Cases:** The total number of confirmed COVID-19 cases.\n
    **Recovered Cases:** The total number of recovered cases from COVID-19.\n
    **Active Cases:** The current number of active cases.\n
    **Recovery Rate:** The ratio of recovered cases to confirmed cases.\n
    **Active Case Rate:** The ratio of active cases to confirmed cases.\n
    **Log Confirmed:** Logarithm of the confirmed cases for normalization.
    """)
    
    # load dataset here and display sample data
    df = pd.read_excel('country_wise_latest.xls')
    st.subheader("Sample Data:")
    st.write(df.head())

# Visualization section
elif navigation == "Visualization":
    st.header("Visualizations")
    
    # Load dataset
    df = pd.read_excel('country_wise_latest.xls')
    
    # Display the summary statistics of the dataset
    st.write('Summary statistics of the dataset:')
    st.write(df.describe())

    # Exclude non-numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Compute the correlation matrix on numeric columns only
    correlation_matrix = numeric_df.corr()

    # Create a correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf()) 

    # Scatterplot: Active Cases vs Confirmed Cases
    st.subheader("Active Cases vs Confirmed Cases")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Confirmed', y='Active')
    plt.title("Active Cases vs Confirmed Cases")
    st.pyplot(plt.gcf())


# Machine Learning Model section
elif navigation == "Machine Learning Model":
    st.header("Machine Learning Model")

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
