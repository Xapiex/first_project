# src/models/predict_model.py
import pathlib
from fastai.vision.all import *

# ¡CRÍTICO! FastAI necesita que 'is_cat' esté en el entorno 
# para poder cargar correctamente el archivo .pkl
from src.features.build_features import is_cat 

def predict(image_path):
    """
    Carga el modelo preentrenado y realiza una predicción sobre una nueva imagen.
    """
    # Resolviendo rutas dinámicamente
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    MODELS_DIR = BASE_DIR / "models"
    model_path = MODELS_DIR / "clasificador.pkl"
    
    if not model_path.exists():
        print(f"Error: No se encontró el modelo en {model_path}.")
        print("Asegúrate de haber ejecutado src/models/train_model.py primero.")
        return
        
    print(f"Cargando modelo empaquetado desde {model_path}...")
    learn = load_learner(model_path)
    
    print(f"Analizando la imagen: {image_path}...")
    # learn.predict devuelve la clase predicha, el índice y un tensor con las probabilidades
    pred_class, pred_idx, probabilities = learn.predict(image_path)
    
    # Extraemos la probabilidad de la clase ganadora y la convertimos a porcentaje
    probabilidad = probabilities[pred_idx].item() * 100

    # El modelo etiqueta a los gatos con True y a los perros con False.
    # Algunos entornos devuelven el valor como bool, otros como string.
    pred_value = pred_class
    if isinstance(pred_value, str):
        pred_value = pred_value.strip().lower() == "true"

    resultado_legible = "GATO" if bool(pred_value) else "PERRO"

    print("\n" + "="*40)
    print("         RESULTADO DE LA PREDICCIÓN")
    print("="*40)
    print(f"  Predicción final : ¡Es un {resultado_legible}!")
    print(f"  Nivel de certeza : {probabilidad:.2f}%")
    print("="*40 + "\n")

    return resultado_legible, probabilidad

if __name__ == "__main__":
    # Bloque de prueba
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    
    # Busca una imagen llamada 'prueba.jpg' en la raíz del proyecto
    ruta_prueba = BASE_DIR / "prueba.jpg"
    
    if ruta_prueba.exists():
        predict(ruta_prueba)
    else:
        print(f"\n[Aviso] Para probar el script, necesitas una imagen.")
        print(f"1. Descarga una foto de un perro o gato.")
        print(f"2. Guárdala como 'prueba.jpg' en la carpeta raíz:")
        print(f"   {BASE_DIR}")
        print(f"3. Vuelve a ejecutar este script.")