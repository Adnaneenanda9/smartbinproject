from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)

# Chargement du modèle
model = load_model(r'C:\Adnane PFA\model.h5', compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dictionnaire pour mapper les indices de classe aux noms de classe
class_dict = {
    0: 'battery',
    1: 'glass', 
    2: 'metal',
    3: 'organic',
    4: 'paper',
    5: 'plastic'
}

def prepare_image(img_bytes):
    try:
        img = load_img(BytesIO(img_bytes), target_size=(224, 224))
        img_array = img_to_array(img)
        # img_array /= 255.0  # Assurez-vous que la mise à l'échelle est correcte
        img_array = np.expand_dims(img_array, axis=0)
        logging.debug(f'Image prétraitée, forme: {img_array.shape}, valeurs: {img_array.min()} à {img_array.max()}')
        return img_array
    except Exception as e:
        logging.error(f'Erreur dans prepare_image: {e}')
        return None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    logging.debug(f"Fichiers dans la requête: {request.files}")
    if 'file' not in request.files:
        logging.error("Aucun fichier trouvé dans la requête")
        return jsonify(error="Aucun fichier fourni"), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("Aucun fichier sélectionné pour l'upload")
        return jsonify(error="Aucun fichier sélectionné"), 400

    img_bytes = file.read()
    logging.debug(f"Taille du fichier reçu: {len(img_bytes)} bytes")
    img = prepare_image(img_bytes)
    
    if img is None:
        logging.error("Erreur de traitement de l'image")
        return jsonify(error="Erreur de traitement de l'image"), 500

    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_dict[predicted_class_index]
    confidence = float(prediction[0][predicted_class_index])

    logging.debug(f'Classe prédite: {predicted_class}, Confiance: {confidence:.3f}')
    return jsonify(predicted_class=predicted_class, confidence=confidence)

@app.route('/test', methods=['GET'])
def test():
    return "La route de test fonctionne !"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')





