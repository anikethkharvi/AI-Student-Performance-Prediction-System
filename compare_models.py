# This file trains and compares multiple machine learning models
# Models used: Logistic Regression, Decision Tree, Random Forest, KNN

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("student_data.csv")

X = data.drop("performance", axis=1)
y = data["performance"]

# Splitting dataset into training and testing sets
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Training the machine learning model
# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Calculating accuracy of the model

accuracies = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {acc*100:.2f}%")

    accuracies.append(acc)

# Plot graph
plt.bar(models.keys(), accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()