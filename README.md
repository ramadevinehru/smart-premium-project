# SmartPremium: Predicting Insurance Costs with Machine Learning

## Overview

**SmartPremium** is a machine learning application designed to predict insurance premiums based on customer characteristics and policy details. Built using Python, Streamlit, and XGBoost, this project aims to assist insurance companies, financial institutions, and healthcare providers in estimating insurance costs accurately.

## Features

- **Data Exploration**: Load and analyze customer data, including age, income, health status, and claim history.
- **Data Preprocessing**: Handle missing values, encode categorical variables, and scale features for optimal model performance.
- **Model Development**: Train and evaluate multiple regression models, including Linear Regression, Decision Trees, Random Forest, and XGBoost.
- **ML Pipeline & MLflow Integration**: Automate the workflow from preprocessing to evaluation and track experiments using MLflow.
- **Interactive Web Application**: Deploy a user-friendly Streamlit app for real-time insurance premium predictions.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Streamlit
- MLflow
- joblib
- requests

## Prerequisites

Ensure you have the following installed:

- Python (>=3.13)
- Required Python libraries (listed in `requirements.txt`)

## Usage
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Interact with the Application**:
   - Input customer details such as age, income, health score, etc.
   - Receive real-time insurance premium predictions.

## Project Structure

- `app.py`: Main application file for the Streamlit app.
- `xgboost_model.pkl`: Trained XGBoost model saved using joblib.
- `train.csv`: Dataset containing customer and policy details.
- `requirements.txt`: List of required Python libraries.

## Contribution

Feel free to contribute by forking the repository and submitting pull requests.

## License

This project is licensed under the MIT License.

## Author

Ramadevi N

