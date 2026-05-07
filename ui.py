# ui.py

import pandas as pd
from model import train_model

# Train model
model = train_model()

print("\nEnter Student Details:")

# User input
math = int(input("Enter math score: "))
reading = int(input("Enter reading score: "))
writing = int(input("Enter writing score: "))
sleep = int(input("Enter sleep hours: "))
stress = int(input("Enter stress level: "))
social = int(input("Enter social media usage: "))

# Calculate average score
average = (math + reading + writing) / 3

# Create DataFrame
new_data = pd.DataFrame(
    [[
        math,
        reading,
        writing,
        average,
        stress,
        sleep,
        social
    ]],
    columns=[
        "math score",
        "reading score",
        "writing score",
        "average_score",
        "stress_level",
        "sleep_hours",
        "social_media_usage"
    ]
)

# Predict
prediction = model.predict(new_data)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Student Performance: GOOD")
else:
    print("Student Performance: POOR")