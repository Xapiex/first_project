# src/features/build_features.py
import pathlib
from fastai.vision.all import *

def is_cat(x):
    """
    Regla de etiquetado: Si el nombre del archivo empieza con mayúscula, es un gato.
    """
    return x[0].isupper()

def get_dataloaders(data_path, img_size=224, valid_pct=0.20, seed=42):
    """
    Toma la ruta de los datos crudos, aplica las transformaciones y 
    construye los DataLoaders para el entrenamiento.
    """
    print(f"Construyendo DataLoaders desde: {data_path}...")
    
    dls = ImageDataLoaders.from_name_func(
        data_path,
        get_image_files(data_path),
        valid_pct=valid_pct,
        seed=seed,
        label_func=is_cat,
        item_tfms=Resize(img_size)
    )
    
    print(f"DataLoaders listos. Clases detectadas: {dls.vocab}")
    return dls

if __name__ == "__main__":
    # Prueba rápida para verificar que los datos se cargan correctamente
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    
    if DATA_DIR.exists():
        dls = get_dataloaders(DATA_DIR)
        print("Prueba de build_features completada con éxito.")
    else:
        print(f"Error: No se encontraron datos en {DATA_DIR}.")
        print("Asegúrate de ejecutar src/data/make_dataset.py primero.")