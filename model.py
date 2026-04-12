# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model():

    print("Model training started...")

    # Load dataset
    data = pd.read_csv("student_data_cleaned.csv")

    # Features and target
    X = data.drop("performance", axis=1)
    y = data["performance"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=2000)

    model.fit(X_train, y_train)

    print("Model trained successfully!")

    return model


if __name__ == "__main__":
    train_model()