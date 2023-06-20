from flask import Flask, render_template, request
import webbrowser
from PIL import Image
import numpy as np
import keras.models
import speech_recognition as sr

app = Flask(__name__)


@app.route("/")
def index1():
    return render_template("index.html")


model = keras.models.load_model("deep1.h5")

# Définition des classes
label_names = [
    "Plane",
    "Car",
    "ird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck",
]


def rechercher_par_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        label_resultat = "Speak Now..."
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio, language="fr-FR,en-US")
        label_resultat = f"{query}"
        global audio_query
        audio_query = query
    except sr.UnknownValueError:
        label_resultat = "Google Speech Recognition did not understand audio"
    except sr.RequestError as e:
        label_resultat = (
            "Unable to connect to Google Speech Recognition service; {0}".format(e)
        )

    return label_resultat


# Fonction de prédiction
def predict(image_path):
    # Chargement de l'image
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img = np.array(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prédiction avec le modèle
    y_pred = model.predict(img)
    y_pred_class = np.argmax(y_pred, axis=1)

    # Récupération de la classe prédite
    predicted_class = label_names[y_pred_class[0]]

    return predicted_class


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" in request.files:
            # Charger l'image
            image = request.files["image"]
            image_path = "static/" + image.filename
            image.save(image_path)

            # Effectuer la prédiction
            predicted_class = predict(image_path)

            return render_template(
                "index.html", image_path=image_path, predicted_class=predicted_class
            )

        elif "audio" in request.files:
            # Charger l'audio et effectuer la recherche
            audio = request.files["audio"]
            audio_query = rechercher_par_audio()

            return render_template("index.html", audio_query=audio_query)

    return render_template("index.html")


@app.route("/search", methods=["POST"])
def web_search():
    query = request.form["query"]
    search_option = request.form["search_option"]

    if search_option == "1":
        url = f"https://www.google.com/search?q={query}&tbm=isch"
    else:
        url = f"https://www.google.com/search?q={query}"

    # Ouvrir la page de recherche dans le navigateur
    webbrowser.open_new(url)

    return render_template("back.html")


if __name__ == "__main__":
    app.run(debug=True)
