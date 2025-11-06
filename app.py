from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

model = keras.models.load_model("./model.keras")
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            daily_time_spent = float(request.form["daily_time_spent"])
            age = float(request.form["age"])
            area_income = float(request.form["area_income"])
            daily_internet_usage = float(request.form["daily_internet_usage"])
            gender = 1 if request.form["gender"] == "Male" else 0

            input_data = np.array([[daily_time_spent, age, area_income, daily_internet_usage, gender]])
            input_data = (input_data - mean) / scale

            prediction = model.predict(input_data)
            result = "Clique sur l'annonce" if prediction[0][0] > 0.5 else "Ne clique pas sur l'annonce"

            return render_template("index.html", prediction=result)

        except Exception as e:
            return render_template("index.html", prediction=f"Erreur : {e}")

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
