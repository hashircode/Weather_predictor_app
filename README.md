 Weather Predictor App

A simple Weather Predictor app is a streamlit application that uses simple mahine learning models to make 7-day weather forecast.It also predicts chances of rain based on parameters of the day.
It also tells us about pressure,temprature,humidity and wind speed .

Screenshot 

![img alt](https://github.com/hashircode/Weather_predictor_app/blob/main/weather%20ss1.png?raw=true)

## Architecture 
![Architecture](https://github.com/hashircode/Weather_predictor_app/blob/09808f96d044fb1150ff23797c697becac8a8b00/arcitectulimage.png)


[![LinkedIn Post Badge](https://img.shields.io/badge/LinkedIn-Post-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/posts/muhammad-hashir-485b53376_im-excited-to-share-the-completion-of-my-activity-7398418334649806848-8WJi)

✨ Features

7-Day Prediction : The application predicts  the weather for the next seven days based on the user's input.

Rain Prediction: An integrated Random Forest model predicts the probability of rain for each day.


🛠️ Technologies Used

User Interface      Streamlit        Creates the interactive, data-driven web application interface.
Backend / Logic     Python           Core application language for running the web app and all logic.
Data Handling       Pandas           Used for creating, manipulating, and managing the weather dataframes (both training and projected data)."
Machine Learning    Scikit-learn     Provides the implementation for the Random Forest Classifier and StandardScaler for the rain prediction model.


Overview

This Streamlit application provides an interactive 7-day weather forecast, driven by user input for the current day's parameters. It leverages an in-memory Random Forest Classifier model to predict the probability of rain across the forecast week. The app serves as a simple demonstration of combining data projection and machine learning for real-time forecasting.

 
🚀 How to Run the Weather Predictor App


Since this application is a single Python script using Streamlit, you do not need a complex Docker setup to run it locally.

Prerequisites

Python (3.7+) installed on your system.

The necessary libraries (streamlit, pandas, scikit-learn) installed.

Step 1: Install Dependencies

Run the following command in your terminal to ensure you have the required libraries:

pip install streamlit pandas scikit-learn


Step 2: Run the Application

Save the provided code as a Python file (e.g., app.py). Then, execute the Streamlit command from your terminal:

streamlit run app.py


Step 3: Access the Dashboard

Once the command runs, Streamlit will automatically open a browser window (or provide a local URL, usually http://localhost:8501).

You can now:

Adjust the input parameters on the left.

View the updated 7-Day Forecast based on the built-in Random Forest model.

## Authors

* **[Muhammad Hashir](https://github.com/account)**
