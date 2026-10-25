
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("iris_model.joblib")


@app.route("/")
def home():
    return jsonify({
        "message": "MLOps Iris Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({
            "error": "Please provide a features list"
        }), 400

    features = data["features"]

    if len(features) != 4:
        return jsonify({
            "error": "Exactly 4 features are required"
        }), 400

    prediction = model.predict([features])[0]

    return jsonify({
        "predicted_class": int(prediction),
        "flower_name": [
            "setosa",
            "versicolor",
            "virginica"
        ][int(prediction)]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
