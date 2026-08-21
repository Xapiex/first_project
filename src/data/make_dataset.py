# src/data/make_dataset.py

import pathlib

from fastai.vision.all import untar_data, URLs


def download_data():
    """
    Descarga el dataset Oxford-IIIT Pets.

    El archivo comprimido se guarda en:
        data/archive/

    El dataset descomprimido se guarda en:
        data/raw/
    """

    # ========================================================
    # RUTA RAÍZ DEL PROYECTO
    # ========================================================

    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

    # Carpeta donde quedará el dataset descomprimido
    DATA_DIR = BASE_DIR / "data" / "raw"

    # Carpeta donde quedará el archivo .tgz descargado
    ARCHIVE_DIR = BASE_DIR / "data" / "archive"

    # Crear carpetas si todavía no existen
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("====================================")
    print("DESCARGA DEL DATASET")
    print("====================================")

    print(f"Dataset: {DATA_DIR}")
    print(f"Archivo comprimido: {ARCHIVE_DIR}")

    # ========================================================
    # DESCARGAR Y DESCOMPRIMIR
    # ========================================================

    path = untar_data(
        URLs.PETS,
        archive=ARCHIVE_DIR,
        data=DATA_DIR
    )

    print("\n====================================")
    print("DATASET LISTO")
    print("====================================")

    print(f"Ruta final: {path}")

    return path


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    download_data()