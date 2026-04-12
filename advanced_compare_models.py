# advanced_compare_models.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------
# Load cleaned dataset
# -----------------------------

data = pd.read_csv("student_data_cleaned.csv")

# Separate features and target
X = data.drop("performance", axis=1)
y = data["performance"]

# Save feature names (for importance graph)
feature_names = X.columns

# -----------------------------
# Split dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Define models
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

accuracies = []

print("\nModel Accuracy Results:\n")

# -----------------------------
# Train and test models
# -----------------------------

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"{name}: {acc*100:.2f}%")

    accuracies.append(acc)

# -----------------------------
# Accuracy Graph
# -----------------------------

plt.figure(figsize=(8, 5))

plt.bar(models.keys(), accuracies)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=30)

plt.tight_layout()

# Save image
plt.savefig("accuracy_graph.png")

plt.show()


# -----------------------------
# Confusion Matrix (Best Model)
# -----------------------------

best_model = LogisticRegression(max_iter=2000)

best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix - Logistic Regression")

# Save image
plt.savefig("confusion_matrix.png")

plt.show()


# -----------------------------
# Feature Importance (Random Forest)
# -----------------------------

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

importances = rf.feature_importances_

plt.figure(figsize=(10, 6))

plt.bar(feature_names, importances)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=90)

plt.tight_layout()

# Save image
plt.savefig("feature_importance.png")

plt.show()


print("\nAll graphs generated and saved successfully!")