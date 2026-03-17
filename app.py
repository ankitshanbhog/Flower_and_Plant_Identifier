from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import io
import os

app = Flask(__name__)

# Load model and classes
model = tf.keras.models.load_model('models/plant_classifier_v1.h5')
with open('models/classes.pkl', 'rb') as f:
    class_names = pickle.load(f)

# --- PLANT INFORMATION DATABASE ---
PLANT_INFO = {
    "aloe_vera_plant": {
        "scientific": "Aloe barbadensis miller",
        "desc": "A succulent known for thick leaves containing a soothing gel used for burns and skincare."
    },
    "curry_leaf_plant": {
        "scientific": "Murraya koenigii",
        "desc": "An aromatic tree native to India; its leaves are a staple in South Asian cuisine and traditional medicine."
    },
    "daisy_flower": {
        "scientific": "Bellis perennis",
        "desc": "A classic European species of the Asteraceae family, recognized by its white petals and yellow center."
    },
    "dandelion_flower": {
        "scientific": "Taraxacum",
        "desc": "A flowering herbaceous perennial often used for its medicinal properties and edible leaves."
    },
    "hibiscus_flower": {
        "scientific": "Hibiscus rosa-sinensis",
        "desc": "Known for large, colorful trumpet-shaped flowers, often used to make herbal teas."
    },
    "lotus_flower": {
        "scientific": "Nelumbo nucifera",
        "desc": "An aquatic perennial representing purity and enlightenment in many Asian cultures."
    },
    "mango_plant": {
        "scientific": "Mangifera indica",
        "desc": "A tropical stone fruit tree. Its leaves are often used in traditional Indian rituals and medicine."
    },
    "marigold_flower": {
        "scientific": "Tagetes",
        "desc": "Bright, hardy flowers often used in festive garlands and known for repelling garden pests."
    },
    "mint_plant": {
        "scientific": "Mentha",
        "desc": "A fast-growing herb used worldwide for its cooling flavor in food, drinks, and essential oils."
    },
    "neem_plant": {
        "scientific": "Azadirachta indica",
        "desc": "A powerful medicinal tree known as 'the village pharmacy' in India for its antibacterial properties."
    },
    "orchid_flower": {
        "scientific": "Orchidaceae",
        "desc": "One of the most diverse flower families, prized globally for their exotic shapes and vibrant colors."
    },
    "rose_flower": {
        "scientific": "Rosa",
        "desc": "The universal symbol of love; a woody perennial known for its multi-layered petals and fragrance."
    },
    "sunflower_flower": {
        "scientific": "Helianthus annuus",
        "desc": "Iconic tall plants with large yellow heads that track the sun across the sky (heliotropism)."
    },
    "tulip_flower": {
        "scientific": "Tulipa",
        "desc": "A spring-blooming bulbous plant that comes in almost every color except pure blue."
    },
    "tulsi_plant": {
        "scientific": "Ocimum tenuiflorum",
        "desc": "Holy Basil is a sacred plant in Hindu culture, used extensively in Ayurveda for respiratory health."
    }
}

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
   
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    img_bytes = file.read()
    processed_img = prepare_image(img_bytes)
    
   
    preds = model.predict(processed_img)
    confidence = float(np.max(preds) * 100)
    result_idx = np.argmax(preds)
    
    
    folder_name = class_names[result_idx] 
    label = folder_name.replace('_', ' ').title()

    
    if confidence < 30.0:
        return jsonify({
            'label': 'Uncertain',
            'confidence': confidence,
            'message': 'Confidence too low (Below 30%). Try a clearer photo.'
        })
    
    
    details = PLANT_INFO.get(folder_name, {
        "scientific": "N/A",
        "desc": "Information not available."
    })
    
    return jsonify({
        'label': label,
        'confidence': confidence,
        'scientific': details['scientific'],
        'desc': details['desc'],
        'message': 'Success'
    })

if __name__ == '__main__':
    app.run(debug=True)