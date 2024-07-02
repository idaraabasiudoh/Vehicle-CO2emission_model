# Vehicle CO2 Emission Prediction

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Contributions](#contributions)
- [Acknowledgments](#acknowledgments)
- [Change Log](#change-log)
- [License](#license)

## Introduction
This repository contains a machine learning project focused on predicting CO2 emissions from various vehicles using a multiple linear regression model. The project leverages Python and popular data science libraries such as scikit-learn, pandas, and matplotlib.

## Objectives
The primary objectives of this project are:
- To implement a linear regression model using scikit-learn to predict CO2 emissions.
- To train, test, and evaluate the model on a dataset of vehicle specifications and CO2 emissions.
- To explore the relationship between different vehicle features and their CO2 emissions.

## Dataset
The dataset used in this project, `FuelConsumptionCo2.csv`, contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. The dataset includes the following columns:
- `MODELYEAR`: e.g., 2014
- `MAKE`: e.g., Acura
- `MODEL`: e.g., ILX
- `VEHICLE CLASS`: e.g., SUV
- `ENGINE SIZE`: e.g., 4.7
- `CYLINDERS`: e.g., 6
- `TRANSMISSION`: e.g., A6
- `FUEL CONSUMPTION in CITY (L/100 km)`: e.g., 9.9
- `FUEL CONSUMPTION in HWY (L/100 km)`: e.g., 8.9
- `FUEL CONSUMPTION COMB (L/100 km)`: e.g., 9.2
- `CO2 EMISSIONS (g/km)`: e.g., 182

[Dataset Source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)

### Requirements

Ensure you have the following dependencies installed:

- `Python 3.x`
- `numpy`
- `pandas`
- `scikit-learn`

## Installation
To run this project locally, you need to have Python installed along with the required libraries. You can install the necessary packages using the following command:

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/idaraabasiudoh/vehicle-co2-emission-prediction.git
cd vehicle-co2-emission-prediction
pip install -r requirements.txt
```

## Usage
To use this repository, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/idaraabasiudoh/vehicle-co2-emission-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd vehicle-co2-emission-prediction
    ```
3. Run the prediction script:
    ```bash
    python predict.py
    ```

## Modeling
The modeling process involves the following steps:
1. **Data Exploration**: Understanding the dataset by visualizing and summarizing the data.
2. **Data Preparation**: Cleaning and splitting the data into training and testing sets.
3. **Model Training**: Using the training set to train a linear regression model.
4. **Model Evaluation**: Evaluating the model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared score.

## Evaluation
The performance of the model is evaluated using the test dataset. The key metrics used for evaluation include:

- **Mean Absolute Error (MAE)**: This is the mean of the absolute value of the errors. It is an easy-to-understand metric representing the average error between the predicted and actual values.
    ```python
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error (MAE):", mae)
    ```

- **Mean Squared Error (MSE)**: This is the mean of the squared errors. It is a popular metric because it gives more weight to larger errors, which can be useful when large errors are particularly undesirable.
    ```python
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)
    ```

- **R-squared Score**: This metric indicates how well the data points fit the regression line. An R-squared score of 1.0 means the model perfectly explains the variance in the target variable.
    ```python
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print("R-squared Score:", r2)
    ```

### Example Code
Here is an example of how to evaluate the model using these metrics:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming y_test contains the actual values and y_pred contains the predicted values
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared Score:", r2)
```

## Contributions
We welcome contributions from the community to improve this project. To contribute, please follow these steps:

1. **Fork the Repository**: Click the "Fork" button at the top right of the repository page to create a copy of this repository on your GitHub account.

2. **Clone the Repository**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/idaraabasiudoh/vehicle-co2-emission-prediction.git
    ```

3. **Create a New Branch**: Create a new branch for your feature or bug fix.
    ```bash
    git checkout -b feature-name
    ```

4. **Make Changes**: Make your changes to the codebase.

5. **Commit Your Changes**: Commit your changes with a clear and descriptive commit message.
    ```bash
    git commit -m "Description of your changes"
    ```

6. **Push to Your Branch**: Push your changes to your forked repository.
    ```bash
    git push origin feature-name
    ```

7. **Open a Pull Request**: Open a pull request to merge your changes into the main repository. Provide a detailed description of your changes in the pull request.

We appreciate your contributions and will review your pull request as soon as possible. Thank you for helping improve this project!

## Acknowledgments 

<a href="http://www.linkedin.com/in/idaraabasiudoh" target="_blank">Idara-Abasi Udoh</a>

Saeed Aghabozorgi


### Other Contributors

<a href="https://www.linkedin.com/in/joseph-s-50398b136/" target="_blank">Joseph Santarcangelo</a>

Azim Hirjani




## Change Log


|  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
|---|---|---|---|
| 2024-07-02 | 2.2 | Idara-Absi Udoh | Project completion|
| 2020-11-03 | 2.1  | Lakshmi Holla  |  Changed URL of the csv |
| 2020-08-27  | 2.0  | Lavanya  |  Moved lab to course repo in GitLab |
|   |   |   |   |

## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
