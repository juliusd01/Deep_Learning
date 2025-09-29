# Sign Prediction Model

This project is designed to predict the sign of stock price movements using historical data. The initial code downloads stock data, calculates log returns, and generates a sign indicator based on those returns. The project will include a model that utilizes this data to make predictions.

## Installation

To set up the project, you will need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Run the `forecast.py` script to download the stock data and prepare the dataset for modeling.

2. **Model Training**: Use the `model.py` script to define and train your sign prediction model.

3. **Making Predictions**: After training, you can use the model to make predictions on new data.