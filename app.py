import io
import os

import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    send_from_directory,
    url_for,
)
from PIL import Image
from keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
try:
    model = load_model("./model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Upload folder for images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Image size
IMAGE_SIZE = (224, 224)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file:
        try:
            # Save the uploaded file
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            # Process the image
            image = Image.open(image_path)
            image = image.resize(IMAGE_SIZE)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Make predictions
            predictions = model.predict(image_array)

            # Process the prediction (assumes classification with softmax)
            predicted_class = np.argmax(predictions, axis=1)[
                0
            ]  # Get the predicted class index

            # If you want to return the image in base64 for display:
            import base64
            from io import BytesIO

            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return render_template(
                "result.html", image_path=img_str, predicted_class=predicted_class
            )

        except Exception as e:
            print(f"Error processing the image: {e}")
            return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
