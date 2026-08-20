# src/data/make_dataset.py
import pathlib
from fastai.vision.data import untar_data

def download_data():
    """
    Descarga el dataset de Oxford-IIIT Pets y lo guarda en la carpeta data/raw/
    en la raíz del proyecto.
    """
    # Calculamos la raíz del proyecto dinámicamente:
    # Este archivo está en src/data/, así que subimos 3 niveles: make_dataset.py -> data -> src -> Raíz
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    
    # Creamos el directorio si no existe
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    DATASET_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz'
    
    print(f"Verificando/Descargando dataset en: {DATA_DIR}...")
    
    # untar_data descarga y descomprime. Redirigimos el destino a nuestra carpeta local.
    path = untar_data(DATASET_URL, dest=DATA_DIR)
    
    print(f"¡Dataset listo y ubicado en: {path}!")
    return path

if __name__ == "__main__":
    download_data()
