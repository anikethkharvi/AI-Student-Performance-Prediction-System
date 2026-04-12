# preprocessing.py

import pandas as pd

def load_data():
    # Load dataset
    data = pd.read_csv("student_data.csv")

    print("Dataset Loaded Successfully!\n")
    print(data.head())

    return data


if __name__ == "__main__":
    load_data()