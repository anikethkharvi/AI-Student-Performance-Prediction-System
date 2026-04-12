# preprocessing.py

import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("StudentsPerformance.csv")

print("Original Dataset Loaded")

# Create average score
data["average_score"] = (
    data["math score"] +
    data["reading score"] +
    data["writing score"]
) / 3

# Add additional features
np.random.seed(42)

data["stress_level"] = np.random.randint(1, 10, len(data))
data["sleep_hours"] = np.random.randint(4, 9, len(data))
data["social_media_usage"] = np.random.randint(1, 6, len(data))

# Create performance column
data["performance"] = (
    (data["average_score"] * 0.5)
    + (data["sleep_hours"] * 5)
    - (data["stress_level"] * 2)
) >= 60

data["performance"] = data["performance"].astype(int)

# Convert categorical values
data = pd.get_dummies(data)

# Save cleaned dataset
data.to_csv("student_data_cleaned.csv", index=False)

print("Preprocessing completed successfully!")