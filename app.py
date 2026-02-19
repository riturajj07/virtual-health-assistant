from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join("..", "models", "health_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "Virtual Health Assistant API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # JSON input from frontend
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
