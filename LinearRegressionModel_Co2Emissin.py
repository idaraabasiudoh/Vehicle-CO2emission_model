import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def model() -> linear_model.LinearRegression:
    """
    Train a linear regression model to predict CO2 emissions based on engine size, 
    number of cylinders, and combined fuel consumption.

    Returns:
        regr (linear_model.LinearRegression): The trained linear regression model.
    """
    # Attempt to read the CSV file containing the dataset
    try:
        df = pd.read_csv("FuelConsumptionCo2.csv")
    except FileNotFoundError:
        print("The file 'FuelConsumptionCo2.csv' was not found.")
        return None

    # Select relevant features and target variable for the model
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

    # Split the data into training and testing sets (80% train, 20% test)
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    # Initialize the linear regression model
    regr = linear_model.LinearRegression()

    # Prepare the training data
    x_train = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    y_train = np.asanyarray(train[['CO2EMISSIONS']])

    # Train the model using the training data
    regr.fit(x_train, y_train)

    # Prepare the testing data
    x_test = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    y_test = np.asanyarray(test[['CO2EMISSIONS']])

    # Make predictions using the testing data
    y_pred = regr.predict(x_test)

    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(y_test, y_pred)

    # Calculate the coefficient of determination (R² score)
    var = (r2_score(y_test, y_pred)) * 100

    # Return the trained model
    return [regr, mse, var]

if __name__ == "__main__":
    # Train the model and store it in the 'model' variable
    model = model()