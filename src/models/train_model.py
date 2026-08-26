# src/models/train_model.py

import os
import time
import pathlib
import argparse

import mlflow

from dotenv import load_dotenv
from fastai.vision.all import *
from torchvision.models import resnet34, efficientnet_b0

from src.features.build_features import get_dataloaders


# ============================================================
# VARIABLES DE ENTORNO
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
# MODELOS DISPONIBLES
# ============================================================

SUPPORTED_MODELS = {
    "resnet34": resnet34,
    "efficientnet_b0": efficientnet_b0
}


# ============================================================
# CONFIGURACIÓN DE MLFLOW
# ============================================================

def configure_mlflow():
    """
    Configura la conexión entre el proyecto
    y el servidor de MLflow.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    print("\n====================================")
    print("MLFLOW CONFIGURADO")
    print("====================================")
    print(
        f"Tracking URI: "
        f"{MLFLOW_TRACKING_URI}"
    )
    print(
        f"Experimento: "
        f"{MLFLOW_EXPERIMENT_NAME}"
    )


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train(
    data_path,
    model_name="resnet34",
    epochs=1,
    img_size=224,
    valid_pct=0.20,
    seed=42,
    batch_size=64
):
    """
    Entrena un modelo de clasificación binaria
    utilizando FastAI.

    Modelos disponibles:

    - resnet34
    - efficientnet_b0

    El entrenamiento se registra automáticamente
    en MLflow.
    """

    # --------------------------------------------------------
    # VALIDAR MODELO
    # --------------------------------------------------------

    if model_name not in SUPPORTED_MODELS:

        raise ValueError(
            f"Modelo no soportado: {model_name}. "
            f"Modelos disponibles: "
            f"{list(SUPPORTED_MODELS.keys())}"
        )

    architecture = SUPPORTED_MODELS[
        model_name
    ]


    # --------------------------------------------------------
    # RUTAS DEL PROYECTO
    # --------------------------------------------------------

    BASE_DIR = (
        pathlib.Path(__file__)
        .resolve()
        .parent
        .parent
        .parent
    )

    MODELS_DIR = (
        BASE_DIR
        / "models"
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # NOMBRE DEL MODELO EXPORTADO
    # --------------------------------------------------------

    model_filename = (
        f"clasificador_"
        f"{model_name}_"
        f"e{epochs}_"
        f"img{img_size}_"
        f"bs{batch_size}.pkl"
    )

    export_path = (
        MODELS_DIR
        / model_filename
    )


    # --------------------------------------------------------
    # CONFIGURAR MLFLOW
    # --------------------------------------------------------

    configure_mlflow()


    # --------------------------------------------------------
    # INFORMACIÓN DEL EXPERIMENTO
    # --------------------------------------------------------

    print("\n====================================")
    print("CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("====================================")

    print(
        f"Modelo: {model_name}"
    )

    print(
        f"Epochs: {epochs}"
    )

    print(
        f"Image size: {img_size}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Validation: {valid_pct}"
    )

    print(
        f"Seed: {seed}"
    )


    # --------------------------------------------------------
    # CREAR DATALOADERS
    # --------------------------------------------------------

    print("\nCargando y procesando datos...")

    dls = get_dataloaders(
        data_path,
        img_size=img_size,
        valid_pct=valid_pct,
        seed=seed,
        batch_size=batch_size
    )


    # --------------------------------------------------------
    # NOMBRE DEL RUN
    # --------------------------------------------------------

    run_name = (
        f"{model_name}_"
        f"e{epochs}_"
        f"img{img_size}_"
        f"bs{batch_size}"
    )


    # --------------------------------------------------------
    # INICIAR RUN DE MLFLOW
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        print("\n====================================")
        print("MLFLOW RUN INICIADO")
        print("====================================")

        print(
            f"Run name: {run_name}"
        )

        print(
            f"Run ID: {run.info.run_id}"
        )


        # ----------------------------------------------------
        # REGISTRAR PARÁMETROS
        # ----------------------------------------------------

        mlflow.log_param(
            "architecture",
            model_name
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
            "batch_size",
            batch_size
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
            "pretrained",
            True
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

        print("\n====================================")
        print("INICIALIZANDO MODELO")
        print("====================================")

        print(
            f"Arquitectura: {model_name}"
        )

        learn = vision_learner(
            dls,
            architecture,
            pretrained=True,
            metrics=[
                accuracy,
                error_rate
            ]
        )


        # ----------------------------------------------------
        # ENTRENAMIENTO
        # ----------------------------------------------------

        print("\n====================================")
        print("INICIANDO ENTRENAMIENTO")
        print("====================================")

        print(
            f"Fine-tuning durante "
            f"{epochs} época(s)..."
        )

        start_time = time.perf_counter()

        learn.fine_tune(
            epochs
        )

        end_time = time.perf_counter()

        training_time = (
            end_time
            - start_time
        )


        # ----------------------------------------------------
        # EVALUACIÓN
        # ----------------------------------------------------

        print("\n====================================")
        print("EVALUANDO MODELO")
        print("====================================")

        results = learn.validate()

        valid_loss = float(
            results[0]
        )

        final_accuracy = float(
            results[1]
        )

        final_error_rate = float(
            results[2]
        )


        # ----------------------------------------------------
        # MOSTRAR MÉTRICAS
        # ----------------------------------------------------

        print(
            f"Validation loss: "
            f"{valid_loss:.6f}"
        )

        print(
            f"Accuracy: "
            f"{final_accuracy:.6f}"
        )

        print(
            f"Accuracy (%): "
            f"{final_accuracy * 100:.2f}%"
        )

        print(
            f"Error rate: "
            f"{final_error_rate:.6f}"
        )

        print(
            f"Error rate (%): "
            f"{final_error_rate * 100:.2f}%"
        )

        print(
            f"Tiempo entrenamiento: "
            f"{training_time:.2f} segundos"
        )


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

        mlflow.log_metric(
            "training_time_seconds",
            training_time
        )


        # ----------------------------------------------------
        # EXPORTAR MODELO
        # ----------------------------------------------------

        print("\n====================================")
        print("GUARDANDO MODELO")
        print("====================================")

        print(
            f"Ruta: {export_path}"
        )

        learn.export(
            export_path
        )


        # ----------------------------------------------------
        # REGISTRAR MODELO COMO ARTIFACT
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
            model_name
        )

        mlflow.set_tag(
            "training_method",
            "transfer-learning"
        )


        # ----------------------------------------------------
        # RESULTADO FINAL
        # ----------------------------------------------------

        print("\n====================================")
        print("ENTRENAMIENTO COMPLETADO")
        print("====================================")

        print(
            f"Modelo: {model_name}"
        )

        print(
            f"Accuracy: "
            f"{final_accuracy * 100:.2f}%"
        )

        print(
            f"Error rate: "
            f"{final_error_rate * 100:.2f}%"
        )

        print(
            f"Validation loss: "
            f"{valid_loss:.6f}"
        )

        print(
            f"Tiempo: "
            f"{training_time:.2f} segundos"
        )

        print(
            f"Run ID: "
            f"{run.info.run_id}"
        )

        print(
            f"Modelo guardado en: "
            f"{export_path}"
        )

        print("====================================")


    return export_path


# ============================================================
# ARGUMENTOS DESDE TERMINAL
# ============================================================

def parse_arguments():
    """
    Permite configurar el entrenamiento
    directamente desde PowerShell.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento del clasificador "
            "Cats vs Dogs con MLflow."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet34",
        choices=[
            "resnet34",
            "efficientnet_b0"
        ],
        help=(
            "Arquitectura del modelo."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help=(
            "Número de épocas de fine-tuning."
        )
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help=(
            "Tamaño de las imágenes."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Número de imágenes por batch."
        )
    )

    parser.add_argument(
        "--valid-pct",
        type=float,
        default=0.20,
        help=(
            "Porcentaje utilizado "
            "para validación."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Semilla para reproducibilidad."
        )
    )

    return parser.parse_args()


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

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

        train(
            data_path=DATA_DIR,
            model_name=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            valid_pct=args.valid_pct,
            seed=args.seed,
            batch_size=args.batch_size
        )

    else:

        print("\n====================================")
        print("ERROR")
        print("====================================")

        print(
            "No se encontraron las imágenes en:"
        )

        print(
            DATA_DIR
        )

        print(
            "\nEjecuta primero:"
        )

        print(
            "python -m src.data.make_dataset"
        )