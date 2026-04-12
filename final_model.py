# final_model.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("student_data_cleaned.csv")

# Features and target
X = data.drop("performance", axis=1)
y = data["performance"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train best model
model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# Save model
with open("final_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Final model saved successfully as final_model.pkl")