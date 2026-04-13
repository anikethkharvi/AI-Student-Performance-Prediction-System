# ui.py

# Import required libraries
import pandas as pd
from model import train_model

# Train the machine learning model
model = train_model()

print("\nEnter Student Details:")

# Taking user input for student features
# These values will be used for prediction

try:
    math = int(input("Enter math score: "))
    reading = int(input("Enter reading score: "))
    writing = int(input("Enter writing score: "))
    sleep = int(input("Enter sleep hours: "))
    stress = int(input("Enter stress level: "))
    social = int(input("Enter social media usage: "))
except ValueError:
    # Handle invalid input (non-numeric values)
    print("Invalid input! Please enter numeric values only.")
    exit()

# Creating a DataFrame from user input
# This matches the format used during model training
new_data = pd.DataFrame(
    [[math, reading, writing, sleep, stress, social]],
    columns=[
        "math score",
        "reading score",
        "writing score",
        "sleep_hours",
        "stress_level",
        "social_media_usage"
    ]
)

# Predict student performance using trained model
prediction = model.predict(new_data)

print("\nPrediction Result:")

# Display result based on prediction output
if prediction[0] == 1:
    print("Student Performance: GOOD")
else:
    print("Student Performance: POOR")