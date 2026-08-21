# src/models/train_model.py

import os
import pathlib
import mlflow

from dotenv import load_dotenv
from fastai.vision.all import *

from src.features.build_features import get_dataloaders


# ============================================================
# CONFIGURACIÓN DE VARIABLES DE ENTORNO
# ============================================================

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "cats-dogs-classifier"
)


# ============================================================
# CONFIGURACIÓN DE MLFLOW
# ============================================================

def configure_mlflow():
    """
    Configura la conexión entre el proyecto y el servidor MLflow.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("MLflow configurado correctamente.")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experimento: {MLFLOW_EXPERIMENT_NAME}")


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train(
    data_path,
    epochs=1,
    img_size=224,
    valid_pct=0.20,
    seed=42
):
    """
    Toma la ruta de los datos, genera los DataLoaders,
    entrena un modelo ResNet34, registra el experimento
    en MLflow y exporta el modelo para inferencia futura.
    """

    # --------------------------------------------------------
    # RUTAS DEL PROYECTO
    # --------------------------------------------------------

    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

    MODELS_DIR = BASE_DIR / "models"

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    export_path = MODELS_DIR / "clasificador.pkl"

    # --------------------------------------------------------
    # CONFIGURAR MLFLOW
    # --------------------------------------------------------

    configure_mlflow()

    # --------------------------------------------------------
    # PREPARAR DATOS
    # --------------------------------------------------------

    print("\nCargando y procesando los datos...")

    dls = get_dataloaders(
        data_path,
        img_size=img_size,
        valid_pct=valid_pct,
        seed=seed
    )

    # --------------------------------------------------------
    # INICIAR RUN DE MLFLOW
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=f"resnet34_{epochs}_epochs"
    ) as run:

        print("\n====================================")
        print("MLFLOW RUN INICIADO")
        print("====================================")

        print(f"Run ID: {run.info.run_id}")

        # ----------------------------------------------------
        # REGISTRAR PARÁMETROS
        # ----------------------------------------------------

        mlflow.log_param(
            "architecture",
            "ResNet34"
        )

        mlflow.log_param(
            "epochs",
            epochs
        )

        mlflow.log_param(
            "image_size",
            img_size
        )

        mlflow.log_param(
            "validation_percentage",
            valid_pct
        )

        mlflow.log_param(
            "seed",
            seed
        )

        mlflow.log_param(
            "framework",
            "FastAI"
        )

        mlflow.log_param(
            "training_images",
            len(dls.train_ds)
        )

        mlflow.log_param(
            "validation_images",
            len(dls.valid_ds)
        )

        # ----------------------------------------------------
        # CREAR MODELO
        # ----------------------------------------------------

        print("\nInicializando el modelo ResNet34...")

        learn = vision_learner(
            dls,
            resnet34,
            metrics=[
                accuracy,
                error_rate
            ]
        )

        # ----------------------------------------------------
        # ENTRENAMIENTO
        # ----------------------------------------------------

        print(
            f"\nIniciando fine-tuning por "
            f"{epochs} época(s)..."
        )

        learn.fine_tune(epochs)

        # ----------------------------------------------------
        # EVALUACIÓN
        # ----------------------------------------------------

        print("\nEvaluando modelo...")

        results = learn.validate()

        valid_loss = float(results[0])
        final_accuracy = float(results[1])
        final_error_rate = float(results[2])

        print(f"Validation loss: {valid_loss:.4f}")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"Error rate: {final_error_rate:.4f}")

        # ----------------------------------------------------
        # REGISTRAR MÉTRICAS EN MLFLOW
        # ----------------------------------------------------

        mlflow.log_metric(
            "valid_loss",
            valid_loss
        )

        mlflow.log_metric(
            "accuracy",
            final_accuracy
        )

        mlflow.log_metric(
            "error_rate",
            final_error_rate
        )

        # ----------------------------------------------------
        # EXPORTAR MODELO FASTAI
        # ----------------------------------------------------

        print(
            f"\nGuardando modelo en: {export_path}"
        )

        learn.export(export_path)

        # ----------------------------------------------------
        # GUARDAR MODELO COMO ARTIFACT EN MLFLOW
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(export_path),
            artifact_path="model"
        )

        # ----------------------------------------------------
        # TAGS
        # ----------------------------------------------------

        mlflow.set_tag(
            "task",
            "binary-image-classification"
        )

        mlflow.set_tag(
            "dataset",
            "Oxford-IIIT Pets"
        )

        mlflow.set_tag(
            "model",
            "ResNet34"
        )

        print("\n====================================")
        print("ENTRENAMIENTO COMPLETADO")
        print("====================================")

        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"Error rate: {final_error_rate:.4f}")
        print(f"Run ID: {run.info.run_id}")

        print("====================================")

    return export_path


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":

    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

    # Utilizamos únicamente las fotografías reales.
    # No incluimos annotations/trimaps porque son máscaras
    # de segmentación y no corresponden a nuestro clasificador.
    DATA_DIR = (
        BASE_DIR
        / "data"
        / "raw"
        / "oxford-iiit-pet"
        / "images"
    )

    if DATA_DIR.exists():

        train(
            DATA_DIR,
            epochs=1
        )

    else:

        print(
            f"Error: No se encontraron las imágenes en {DATA_DIR}."
        )

        print(
            "Asegúrate de ejecutar "
            "src/data/make_dataset.py primero."
        )