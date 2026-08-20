# src/visualization/visualize.py
import pathlib
import matplotlib.pyplot as plt
from fastai.vision.all import *

# Importaciones necesarias para cargar los datos y el modelo correctamente
from src.features.build_features import is_cat, get_dataloaders

def show_data_batch(data_path):
    """
    Carga un lote (batch) de imágenes y las muestra con sus etiquetas.
    Excelente para verificar que el dataset se haya cargado bien.
    """
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    
    print(f"Cargando datos desde {data_path}...")
    dls = get_dataloaders(data_path)
    
    print("Generando visualización del lote de datos...")
    # show_batch prepara el gráfico en memoria
    dls.show_batch(max_n=9)
    
    # plt.show() fuerza a que la ventana con el gráfico se abra en tu sistema operativo
    plt.show()

def plot_confusion_matrix():
    """
    Carga el modelo entrenado y los datos para generar una matriz de confusión
    sobre el conjunto de validación.
    """
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    model_path = BASE_DIR / "models" / "clasificador.pkl"
    data_path = BASE_DIR / "data" / "raw"
    
    if not model_path.exists():
        print(f"Error: No se encontró el modelo en {model_path}.")
        print("Debes ejecutar src/models/train_model.py primero.")
        return
        
    if not data_path.exists():
        print("Error: No se encontraron los datos crudos.")
        return

    print("Cargando el modelo empaquetado...")
    learn = load_learner(model_path)
    
    print("Cargando datos de validación...")
    # Necesitamos reconstruir los DataLoaders para tener el set de validación
    dls = get_dataloaders(data_path)
    
    # Le inyectamos los DataLoaders al modelo cargado
    learn.dls = dls
    
    print("Calculando predicciones para la matriz de confusión...")
    # ClassificationInterpretation es una herramienta de FastAI para evaluar modelos
    interp = ClassificationInterpretation.from_learner(learn)
    
    print("Generando gráfico...")
    interp.plot_confusion_matrix(figsize=(6,6), dpi=100)
    plt.show()

if __name__ == "__main__":
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    
    print("=== MÓDULO DE VISUALIZACIÓN ===")
    print("1. Ver lote de imágenes de entrenamiento")
    print("2. Ver matriz de confusión del modelo entrenado")
    opcion = input("Elige una opción (1 o 2): ")
    
    if opcion == '1':
        if DATA_DIR.exists():
            show_data_batch(DATA_DIR)
        else:
            print("Datos no encontrados. Ejecuta make_dataset.py primero.")
    elif opcion == '2':
        plot_confusion_matrix()
    else:
        print("Opción no válida.")