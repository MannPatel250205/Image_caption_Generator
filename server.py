from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import pickle
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# Global variables
model = None
tokenizer = None

def load():
    try:
        model = load_model("/artifacts/best_model.keras")
        with open("/artifacts/tokenize.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise

def idx_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

def feature_extraction(image):
    try:
        vgg_model = VGG16(weights="imagenet")
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = vgg_model.predict(image, verbose=0)
        return feature
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise

def predict_caption(model, image, tokenizer, max_length):
    try:
        in_text = "startseq"
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], max_length, padding="post")
            yhat = model.predict([image, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = idx_to_word(yhat, tokenizer)
            if word is None:
                break
            in_text += " " + word
            if word == "endseq":
                break
        return in_text
    except Exception as e:
        print(f"Error during caption generation: {e}")
        raise

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        image = Image.open(file)
        image_features = feature_extraction(image)
        captions = predict_caption(model, image_features, tokenizer, 35)
        captions = captions.replace("startseq ", "").replace(" endseq", "")
        return jsonify({"caption": captions})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Failed to generate caption"}), 500

if __name__ == "__main__":
    model, tokenizer = load()
    app.run(debug=True)