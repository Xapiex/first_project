# src/features/build_features.py

import pathlib
from fastai.vision.all import *


def is_cat(filename):
    """
    Regla de etiquetado del dataset Oxford-IIIT Pets.

    Si el nombre del archivo comienza con mayúscula,
    corresponde a un gato.

    True  -> Gato
    False -> Perro
    """
    return filename[0].isupper()


def get_dataloaders(
    data_path,
    img_size=224,
    valid_pct=0.20,
    seed=42,
    batch_size=64
):
    """
    Construye los DataLoaders utilizados para entrenamiento
    y validación.

    Parameters
    ----------
    data_path:
        Ruta a las imágenes.

    img_size:
        Tamaño al que se redimensionarán las imágenes.

    valid_pct:
        Porcentaje utilizado para validación.

    seed:
        Semilla para reproducibilidad.

    batch_size:
        Número de imágenes por lote.
    """

    print(
        f"Construyendo DataLoaders desde: {data_path}..."
    )

    files = get_image_files(data_path)

    print(
        f"Imágenes encontradas: {len(files)}"
    )

    dls = ImageDataLoaders.from_name_func(
        data_path,
        files,
        valid_pct=valid_pct,
        seed=seed,
        label_func=is_cat,
        item_tfms=Resize(img_size),

        # Tamaño de cada batch
        bs=batch_size,

        # IMPORTANTE PARA WINDOWS
        # Evita problemas de multiprocessing/pickle
        num_workers=0
    )

    print(
        f"DataLoaders listos. "
        f"Clases detectadas: {dls.vocab}"
    )

    print(
        f"Train: {len(dls.train_ds)}"
    )

    print(
        f"Valid: {len(dls.valid_ds)}"
    )

    return dls


# ============================================================
# PRUEBA DIRECTA
# ============================================================

if __name__ == "__main__":

    BASE_DIR = (
        pathlib.Path(__file__)
        .resolve()
        .parent
        .parent
        .parent
    )

    DATA_DIR = (
        BASE_DIR
        / "data"
        / "raw"
        / "oxford-iiit-pet"
        / "images"
    )

    if DATA_DIR.exists():

        dls = get_dataloaders(DATA_DIR)

        print(
            "Prueba de build_features "
            "completada correctamente."
        )

    else:

        print(
            f"Error: no se encontraron imágenes en:"
        )

        print(DATA_DIR)