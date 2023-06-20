import tkinter as tk
import webbrowser
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import keras.models
import speech_recognition as sr

model = keras.models.load_model('deep1.h5')

# Définition des classes
label_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

def rechercher_par_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        label_resultat.configure(text="Parlez maintenant...", anchor='center', justify='center')
        window.update()
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio, language='fr-FR')
        label_resultat.configure(text=f"Vous avez dit: {query}", anchor='center', justify='center')  # Update the label with the recognized query
        window.update()
        global audio_query
        audio_query = query  # Store the recognized query for later use
    except sr.UnknownValueError:
        label_resultat.configure(text="Google Speech Recognition n'a pas compris l'audio", anchor='center', justify='center')
        window.update()
    except sr.RequestError as e:
        label_resultat.configure(text="Impossible de se connecter à Google Speech Recognition service; {0}".format(e), anchor='center', justify='center')
        window.update()


# Fonction de prédiction
def predict(image_path):
    # Chargement de l'image
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Prédiction avec le modèle
    y_pred = model.predict(img)
    y_pred_class = np.argmax(y_pred, axis=1)

    # Affichage de la prédiction
    label_resultat.configure(text=label_names[y_pred_class[0]])

    # Stockage de la classe prédite pour la recherche sur Google
    global predicted_class
    predicted_class = label_names[y_pred_class[0]]

# Fonction pour charger une image, effectuer la prédiction et conserver l'image
def upload_image():
    # Ouverture de la boîte de dialogue pour sélectionner un fichier
    path = filedialog.askopenfilename()

    if path:
        # Affichage de l'image dans la fenêtre
        img = Image.open(path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        canvas_image.create_image(0, 0, anchor='nw', image=img_tk)
        canvas_image.image = img_tk

        # Prédiction avec le modèle
        predict(path)

        # Stockage de l'image chargée
        global uploaded_image
        uploaded_image = img

def upload_audio():
    rechercher_par_audio()

# Fonction pour effectuer la recherche sur Google
def web_search():
    if uploaded_image is not None and audio_query:
        if var_option.get() == 1:
            url = f'https://www.google.com/search?q={predicted_class}&tbm=isch'
        else:
            url = f'https://www.google.com/search?q={predicted_class}'
    elif uploaded_image is not None:
        if var_option.get() == 1:
            url = f'https://www.google.com/search?q={predicted_class}&tbm=isch'
        else:
            url = f'https://www.google.com/search?q={predicted_class}'
    elif audio_query:
        url = f'https://www.google.com/search?q={audio_query}'
    else:
        print("Aucune image ou audio chargé.")

    # Ouverture de la page de recherche dans le navigateur
    webbrowser.open_new(url)

# Création de la fenêtre
window = tk.Tk()
window.title('Moteur de recherche Noan')
window.geometry('500x550')
window.resizable(False, False)

# Bouton pour charger une image
button_upload_image = tk.Button(window, text="Uploader et predict l'image", command=upload_image)
button_upload_image.pack(pady=10)

button_upload_audio = tk.Button(window, text='Enregistrer un audio', command=upload_audio)
button_upload_audio.pack(pady=10)

# Affichage de l'image sélectionnée
canvas_image = tk.Canvas(window, width=300, height=300)
canvas_image.pack()

# Label pour afficher le résultat de la prédiction
label_resultat = tk.Label(window, font=('Arial', 24))
label_resultat.pack()

# Bouton pour la recherche sur Google
button_web_search = tk.Button(window, text='Recherche sur Google', command=web_search)
button_web_search.pack(pady=10)

# Option pour la recherche sur Google
var_option = tk.IntVar(value=1)
radio_images = tk.Radiobutton(window, text='Recherche d\'images', variable=var_option, value=1)
radio_images.pack()
radio_sites = tk.Radiobutton(window, text='Recherche de sites', variable=var_option, value=2)
radio_sites.pack()

# Variables globales
uploaded_image = None
predicted_class = ""
audio_query = ""

# Boucle principale de la fenêtre
window.mainloop()
