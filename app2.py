from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from keras.models import load_model
import cv2
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import backend as K  # Import para liberar memoria

names = [
    'Amazona Alinaranja', 'Amazona de San Vicente', 'Amazona Mercenaria', 'Amazona Real',
    'Aratinga de Pinceles', 'Aratinga de Wagler', 'Aratinga Ojiblanca', 'Aratinga Orejigualda',
    'Aratinga Pertinaz', 'Batará Barrado', 'Batará Crestibarrado', 'Batara Crestinegro',
    'Batará Mayor', 'Batará Pizarroso Occidental', 'Batará Unicolor', 'Cacatua Ninfa',
    'Catita Frentirrufa', 'Cotorra Colinegra', 'Cotorra Pechiparda', 'Cotorrita Alipinta',
    'Cotorrita de Anteojos', 'Guacamaya Roja', 'Guacamaya Verde', 'Guacamayo Aliverde',
    'Guacamayo azuliamarillo', 'Guacamayo Severo', 'Hormiguerito Coicorita Norteño',
    'Hormiguerito Coicorita Sureño', 'Hormiguerito Flanquialbo', 'Hormiguerito Leonado',
    'Hormiguerito Plomizo', 'Hormiguero Azabache', 'Hormiguero Cantor', 'Hormiguero de Parker',
    'Hormiguero Dorsicastaño', 'Hormiguero Guardarribera Oriental', 'Hormiguero Inmaculado',
    'Hormiguero Sencillo', 'Hormiguero Ventriblanco', 'Lorito Amazonico', 'Lorito Cabecigualdo',
    'Lorito de fuertes', 'Loro Alibronceado', 'Loro Cabeciazul', 'Loro Cachetes Amarillos',
    'Loro Corona Azul', 'Loro Tumultuoso', 'Ojodefuego Occidental', 'Periquito Alas Amarillas',
    'Periquito Australiano', 'Periquito Barrado', 'Tiluchí Colilargo', 'Tiluchí de Santander',
    'Tiluchi Lomirrufo'
]

# Inicializa la app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://api-for-birds-classification.onrender.com/"}})

# Configuración para Render
app.config['UPLOAD_FOLDER'] = '/tmp/uploaded_images'  # Carpeta temporal
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Cargar el modelo una vez para evitar cargarlo múltiples veces
model_path = os.path.join(os.path.dirname(__file__), 'model_VGG16_v4.keras')
modelt = load_model(model_path)

# Endpoint para subir y clasificar la imagen
@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    # Verificar si hay un archivo en la solicitud
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Guardar el archivo
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # Leer, redimensionar y preprocesar la imagen
            imaget = cv2.resize(cv2.imread(filepath), (224, 224), interpolation=cv2.INTER_AREA)
            xt = np.expand_dims(preprocess_input(np.asarray(imaget)), axis=0)

            # Obtener las predicciones
            preds = modelt.predict(xt)
            predicted_class_index = np.argmax(preds)
            predicted_class_name = names[predicted_class_index]
            confidence_percentage = preds[0][predicted_class_index] * 100

            # Limpiar la imagen después de usarla
            os.remove(filepath)
            K.clear_session()  

            return jsonify({
                "message": f'Clase predicha: {predicted_class_name}, Porcentaje de confianza: {confidence_percentage:.2f}%',
            }), 200

        except Exception as e:
            # En caso de error, eliminar el archivo y devolver el error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Error processing the image: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid file type"}), 400

# Verificación de tipo de archivo
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Error 404 personalizado
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Recurso no encontrado"}), 404

# Ruta para servir la interfaz
@app.route('/')
def serve_interface():
    return send_from_directory('.', 'index2.html')

# Ejecución de la aplicación
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port, debug=True)
