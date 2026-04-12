# ui.py

import pandas as pd
from model import train_model

# Train model
model = train_model()

print("\nEnter Student Details:")

math = int(input("Enter math score: "))
reading = int(input("Enter reading score: "))
writing = int(input("Enter writing score: "))
sleep = int(input("Enter sleep hours: "))
stress = int(input("Enter stress level: "))
social = int(input("Enter social media usage: "))

# Create DataFrame
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

prediction = model.predict(new_data)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Student Performance: GOOD")
else:
    print("Student Performance: POOR")