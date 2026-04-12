# ui.py

import pandas as pd
from model import train_model

# Train model
model = train_model()

print("\nEnter Student Details:")

attendance = int(input("Enter attendance (%): "))
study_hours = int(input("Enter study hours per day: "))
previous_marks = int(input("Enter previous marks: "))

# Create DataFrame (fix warning)
new_data = pd.DataFrame(
    [[attendance, study_hours, previous_marks]],
    columns=["attendance", "study_hours", "previous_marks"]
)

# Prediction
prediction = model.predict(new_data)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Student Performance: GOOD")
else:
    print("Student Performance: POOR")