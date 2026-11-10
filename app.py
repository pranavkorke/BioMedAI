
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

SIZE = 224
MODEL_PATH = "biomedai_cnn.keras"

app = Flask(__name__, static_folder="static")

model = None

@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.post("/predict")
def predict():
    global model

    try:
        if model is None:
            model = tf.keras.models.load_model(MODEL_PATH)

        if "image" not in request.files:
            return jsonify(error="No image uploaded"), 400

        file = request.files["image"]

        img = Image.open(io.BytesIO(file.read()))
        img = img.convert("L").resize((SIZE, SIZE))

        x = np.asarray(img, dtype="float32") / 255.0
        x = x[None, ..., None]

        score = float(model.predict(x, verbose=0)[0][0])

        return jsonify(
            score=score,
            prediction="Pneumonia" if score >= 0.5 else "Normal",
            x=0.5,
            y=0.5
        )

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True)
