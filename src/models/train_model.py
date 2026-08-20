# src/models/train_model.py
import pathlib
from fastai.vision.all import *

# Importamos la función que construye los DataLoaders desde nuestro módulo de features
from src.features.build_features import get_dataloaders

def train(data_path, epochs=1):
    """
    Toma la ruta de los datos, genera los DataLoaders, entrena 
    un modelo ResNet34 y lo exporta para inferencia futura.
    """
    # Calculamos dinámicamente las rutas
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    MODELS_DIR = BASE_DIR / "models"
    
    # Nos aseguramos de que la carpeta models/ exista
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export_path = MODELS_DIR / "clasificador.pkl"

    print("Cargando y procesando los datos...")
    dls = get_dataloaders(data_path)
    
    print("Inicializando el modelo ResNet34...")
    # vision_learner descarga los pesos preentrenados y adapta la última capa
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    
    print(f"Iniciando el ajuste fino (fine-tuning) por {epochs} época(s)...")
    learn.fine_tune(epochs)
    
    print(f"Guardando el modelo empaquetado en: {export_path}")
    learn.export(export_path)
    print("¡Entrenamiento y exportación completados con éxito!")
    
    return export_path

if __name__ == "__main__":
    # Bloque de ejecución principal para probar el entrenamiento
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    
    if DATA_DIR.exists():
        # Entrenamos con 1 época por defecto para pruebas rápidas
        train(DATA_DIR, epochs=1)
    else:
        print(f"Error: No se encontraron datos en {DATA_DIR}.")
        print("Asegúrate de ejecutar src/data/make_dataset.py primero.")