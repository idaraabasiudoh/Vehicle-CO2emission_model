#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import linear_model

# Read the data
df = pd.read_csv("FuelConsumptionCo2.csv")

# Select relevant features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Split data into train and test sets
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Using FUELCONSUMPTION_COMB as the feature for training and testing
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

print("Model trained successfully.")
print("Coefficient: ", regr.coef_[0][0])
print("Intercept: ", regr.intercept_[0])

# Function to predict CO2 emissions based on user input
def predict_co2_emissions(fuel_consumption):
    fuel_consumption = np.array([[fuel_consumption]])
    prediction = regr.predict(fuel_consumption)
    return prediction[0][0]

# Get user input and predict
try:
    user_input = float(input("Enter fuel consumption (in L/100km): "))
    predicted_emission = predict_co2_emissions(user_input)
    print(f"Predicted CO2 emission for fuel consumption {user_input} L/100km is: {predicted_emission:.2f} g/km")
except ValueError:
    print("Invalid input. Please enter a numerical value.")