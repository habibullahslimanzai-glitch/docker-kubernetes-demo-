
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save the trained model
joblib.dump(model, "iris_model.joblib")

# Save the evaluation results
results = {
    "model": "RandomForestClassifier",
    "dataset": "Iris",
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "accuracy": round(accuracy, 4)
}

with open("results.json", "w") as file:
    json.dump(results, file, indent=4)

print("Model training completed!")
print("Model accuracy:", round(accuracy * 100, 2), "%")
print("Model saved: iris_model.joblib")
print("Results saved: results.json")
