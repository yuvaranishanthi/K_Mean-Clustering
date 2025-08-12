from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Paths to model files
MODEL_DIR = "model"
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
CLUSTER_DESC_PATH = os.path.join(MODEL_DIR, "cluster_descriptions.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")
FEATURE_RANGES_PATH = os.path.join(MODEL_DIR, "feature_ranges.pkl")

# ✅ Load with joblib (matches how we saved it)
kmeans = joblib.load(KMEANS_MODEL_PATH)
cluster_descriptions = joblib.load(CLUSTER_DESC_PATH)
features = joblib.load(FEATURES_PATH)
feature_ranges = joblib.load(FEATURE_RANGES_PATH)

# Step 1 & Step 2 combined into one page
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None

    if request.method == "POST":
        try:
            # Collect feature values from form
            input_data = [float(request.form[feature]) for feature in features]
            input_array = np.array(input_data).reshape(1, -1)

            # Predict cluster
            prediction = kmeans.predict(input_array)[0]
            prediction_text = cluster_descriptions.get(prediction, f"Cluster {prediction}")

        except Exception as e:
            prediction_text = f"Error: {e}"

        # Show result page
        return render_template("result.html", prediction=prediction_text)

    # Initial GET request → show the form
    return render_template("step.html", features=features, feature_ranges=feature_ranges)


if __name__ == "__main__":
    app.run(debug=True)
