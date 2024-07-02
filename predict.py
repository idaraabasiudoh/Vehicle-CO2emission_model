import tkinter as tk
from tkinter import messagebox
import numpy as np
from LinearRegressionModel_Co2Emissin import model

regr, mse, var = model()[0], model()[1], model()[2]

# Function to predict CO2 emissions based on user input
def predict_emissions():
    try:
        engine_size = float(engine_size_entry.get())
        cylinders = int(cylinders_entry.get())
        fuel_consumption_comb = float(fuel_consumption_comb_entry.get())
        
        # Prepare input data for prediction
        input_data = np.array([[engine_size, cylinders, fuel_consumption_comb]])
        
        # Make prediction
        prediction = regr.predict(input_data)
        
        # Display prediction
        result_label.config(text=f"Predicted CO2 Emissions: {prediction[0][0]:.2f}")
        
        
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid input values.")

# Initialize the main window
root = tk.Tk()
root.title("CO2 Emission Prediction")
root.resizable(False, False)

# Create and place labels and entry widgets for user input
tk.Label(root, text="Engine Size:").grid(row=0, column=0, padx=10, pady=5)
engine_size_entry = tk.Entry(root)
engine_size_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Cylinders:").grid(row=1, column=0, padx=10, pady=5)
cylinders_entry = tk.Entry(root)
cylinders_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Fuel Consumption (Combined):").grid(row=2, column=0, padx=10, pady=5)
fuel_consumption_comb_entry = tk.Entry(root)
fuel_consumption_comb_entry.grid(row=2, column=1, padx=10, pady=5)

# Create and place a button to trigger prediction
predict_button = tk.Button(root, text="Predict CO2 Emissions", command=predict_emissions)
predict_button.grid(row=3, column=0, columnspan=2, pady=10)

# Create and place labels to display results
result_label = tk.Label(root, text="Predicted CO2 Emissions: N/A")
result_label.grid(row=4, column=0, columnspan=2, pady=10)

mse_label = tk.Label(root, text=f"Mean Squared Error: {mse:.2f}")
mse_label.grid(row=7, column=0, columnspan=2, pady=5)

var_label = tk.Label(root, text=f"Model Score: {var:.2f}%")
var_label.grid(row=8, column=0, columnspan=2, pady=5)

# Start the Tkinter main loop
root.mainloop()