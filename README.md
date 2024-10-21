# Stock Price Prediction using LSTM

This repository contains a project for predicting stock prices using Long Short-Term Memory (LSTM) networks. The project utilizes historical stock data to forecast future prices using deep learning techniques. 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Results](#results)
- [Challenges](#challenges)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Introduction
Predicting stock prices is a challenging task due to the volatile and complex nature of financial markets. This project aims to use LSTM networks, a type of recurrent neural network, to model the temporal dependencies in stock price data and make accurate predictions.

## Features
- **Data Preprocessing**: Clean and preprocess stock price data.
- **Feature Engineering**: Select and engineer features for model training.
- **LSTM Model Implementation**: Build and train LSTM models for time series forecasting.
- **Hyperparameter Tuning**: Optimize model parameters for better performance.
- **Visualization**: Plot stock prices and prediction results for analysis.
- **Web Application**: A web interface for real-time stock price prediction.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/stock_price_prediction_lstm.git
   cd stock_price_prediction_lstm

2. Install the required packages in the requirements.txt file:
   
       pip install -r requirements.txt

## Usage

1) Else to run the the project form terminal type in this on your directory:

       streamlit run web_app.py

2) If you wish to train the entire model from strach then do the following steps:

    - **Data Preparation**: Ensure your dataset is formatted correctly in Stock_Symbols.csv
    - **Model Preparation**: Open the Jupter Notebook (**best_model_features.ipynb**) and run it and once it is completed you will           have your final model details.
    - **Train the best model**: In the (**final_model.ipynb**) input all the details and run the notebook. Change the model name so          that you dont match with mine. Once it is saved input the model (**your_model.keras**) in the (**web_app.py**) file.
    - **Run the web app**: Open the terminal and then open your directory then run the following command:

          streamlit run web_app.py

## Project Structure

The project structure is as follows:

    ├── README.md
    ├── requirements.txt
    ├── Stock_Symbols.csv
    ├── best_model_features.ipynb
    ├── final_model.ipynb
    ├── my_model.keras
    ├── web_app.py

- **README.md**: Project documentation.
- **requirements.txt**: List of dependencies.
- **Stock_Symbols.csv**: Dataset of stock symbols and their historical prices.
- **best_model_features.ipynb**: Notebook for feature selection and model optimization.
- **final_model.ipynb**: Notebook for training the final LSTM model.
- **my_model.keras**: Saved trained model.
- **web_app.py**: Web application script.


## Data Preparation

Data preparation involves cleaning and preprocessing the raw stock price data. This includes handling missing values, normalizing data, and creating time series features for the LSTM model.

## Model Training

1) **Feature Engineering**: Use best_model_features.ipynb to preprocess data and select features.
2) **Model Training**: Train the LSTM model using final_model.ipynb. This notebook includes steps for data preparation, model            training and evaluation.

## Results

The model achieves an accuracy of over 94% in predicting stock prices. Detailed results and visualizations are provided in the notebooks, showcasing the performance of the LSTM model.

## Challenges

- **Data Quality**: Handling missing and noisy data.
- **Model Complexity**: Tuning hyperparameters and optimizing the LSTM model.
- **Computational Resources**: Training deep learning models requires significant computational power.

## Future Work

- **Enhance Model**: Experiment with different neural network architectures and techniques.
- **Expand Dataset**: Use larger and more diverse datasets to improve model generalization.
- **Real-time Data**: Integrate real-time stock data for live predictions.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.













