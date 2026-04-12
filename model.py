# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def train_model():
    # Load dataset
    data = pd.read_csv("student_data.csv")

    # Features (Input)
    X = data[["attendance", "study_hours", "previous_marks"]]

    # Target (Output)
    y = data["performance"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Model trained successfully!")

    return model


if __name__ == "__main__":
    train_model()