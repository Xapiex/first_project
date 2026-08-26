# src/models/evaluate_model.py

import os
import argparse
import pathlib

import torch
import matplotlib.pyplot as plt
import mlflow

from dotenv import load_dotenv
from fastai.vision.all import load_learner

from src.features.build_features import get_dataloaders


# ============================================================
# CONFIGURACIÓN MLFLOW
# ============================================================

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def generate_confusion_matrix(
    model_path,
    model_name,
    data_path,
    img_size=224,
    valid_pct=0.20,
    seed=42,
    batch_size=64,
    run_id=None
):
    """
    Carga un modelo FastAI ya entrenado y genera
    su matriz de confusión utilizando el mismo
    conjunto de validación utilizado durante
    el entrenamiento.
    """

    print("\n====================================")
    print("EVALUACIÓN DEL MODELO")
    print("====================================")

    print(f"Modelo: {model_name}")
    print(f"Archivo: {model_path}")

    model_path = pathlib.Path(model_path)
    data_path = pathlib.Path(data_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{model_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset:\n{data_path}"
        )


    # --------------------------------------------------------
    # RECONSTRUIR PARTICIÓN TRAIN / VALID
    # --------------------------------------------------------

    print("\nReconstruyendo conjunto de validación...")

    split_dls = get_dataloaders(
        data_path,
        img_size=img_size,
        valid_pct=valid_pct,
        seed=seed,
        batch_size=batch_size
    )

    valid_items = list(
        split_dls.valid_ds.items
    )

    print(
        f"Imágenes de validación: "
        f"{len(valid_items)}"
    )


    # --------------------------------------------------------
    # CARGAR MODELO
    # --------------------------------------------------------

    print("\nCargando modelo entrenado...")

    learn = load_learner(
        model_path,
        cpu=True
    )


    # --------------------------------------------------------
    # CREAR DATALOADER DE VALIDACIÓN
    #
    # Utilizamos las transformaciones almacenadas
    # dentro del modelo exportado.
    # --------------------------------------------------------

    valid_dl = learn.dls.test_dl(
        valid_items,
        with_labels=True,
        bs=batch_size,
        num_workers=0
    )


    # --------------------------------------------------------
    # PREDICCIONES
    # --------------------------------------------------------

    print("\nGenerando predicciones...")

    predictions, targets = learn.get_preds(
        dl=valid_dl
    )

    predicted_classes = predictions.argmax(
        dim=1
    )


    # --------------------------------------------------------
    # MATRIZ 2x2
    # --------------------------------------------------------

    confusion_matrix = torch.zeros(
        (2, 2),
        dtype=torch.int64
    )

    for real, predicted in zip(
        targets,
        predicted_classes
    ):

        confusion_matrix[
            int(real),
            int(predicted)
        ] += 1


    cm = confusion_matrix.numpy()

    print("\nMatriz de confusión:")

    print(cm)


    # --------------------------------------------------------
    # CLASES
    #
    # False = Perro
    # True  = Gato
    # --------------------------------------------------------

    class_names = [
        "Perro",
        "Gato"
    ]


    # --------------------------------------------------------
    # CREAR GRÁFICA
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        cm,
        cmap="Blues"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_xticks(
        range(len(class_names))
    )

    ax.set_yticks(
        range(len(class_names))
    )

    ax.set_xticklabels(
        class_names
    )

    ax.set_yticklabels(
        class_names
    )

    ax.set_xlabel(
        "Clase predicha"
    )

    ax.set_ylabel(
        "Clase real"
    )

    ax.set_title(
        f"Matriz de confusión - {model_name}"
    )


    # --------------------------------------------------------
    # ESCRIBIR NÚMEROS
    # --------------------------------------------------------

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )


    fig.tight_layout()


    # --------------------------------------------------------
    # GUARDAR FIGURA
    # --------------------------------------------------------

    BASE_DIR = (
        pathlib.Path(__file__)
        .resolve()
        .parent
        .parent
        .parent
    )

    FIGURES_DIR = (
        BASE_DIR
        / "reports"
        / "figures"
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FIGURES_DIR
        / f"confusion_matrix_{model_name}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"\nGráfica guardada en:\n"
        f"{output_path}"
    )


    # --------------------------------------------------------
    # RESULTADOS INDIVIDUALES
    # --------------------------------------------------------

    dog_correct = int(cm[0, 0])
    dog_as_cat = int(cm[0, 1])

    cat_as_dog = int(cm[1, 0])
    cat_correct = int(cm[1, 1])


    print("\n====================================")
    print("RESULTADOS")
    print("====================================")

    print(
        f"Perros clasificados correctamente: "
        f"{dog_correct}"
    )

    print(
        f"Perros clasificados como gatos: "
        f"{dog_as_cat}"
    )

    print(
        f"Gatos clasificados como perros: "
        f"{cat_as_dog}"
    )

    print(
        f"Gatos clasificados correctamente: "
        f"{cat_correct}"
    )


    # --------------------------------------------------------
    # GUARDAR EN MLFLOW
    # --------------------------------------------------------

    if run_id:

        print("\nRegistrando matriz en MLflow...")

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        with mlflow.start_run(
            run_id=run_id
        ):

            mlflow.log_artifact(
                str(output_path),
                artifact_path="evaluation"
            )

            mlflow.log_metric(
                "dog_correct",
                dog_correct
            )

            mlflow.log_metric(
                "dog_as_cat",
                dog_as_cat
            )

            mlflow.log_metric(
                "cat_as_dog",
                cat_as_dog
            )

            mlflow.log_metric(
                "cat_correct",
                cat_correct
            )

        print(
            "Matriz registrada en el Run "
            "existente de MLflow."
        )


    return cm


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Generar matriz de confusión "
            "de un modelo entrenado."
        )
    )

    parser.add_argument(
        "--model-path",
        required=True
    )

    parser.add_argument(
        "--model-name",
        required=True
    )

    parser.add_argument(
        "--run-id",
        default=None
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--valid-pct",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


# ============================================================
# MAIN
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

    generate_confusion_matrix(
        model_path=args.model_path,
        model_name=args.model_name,
        data_path=DATA_DIR,
        img_size=args.img_size,
        valid_pct=args.valid_pct,
        seed=args.seed,
        batch_size=args.batch_size,
        run_id=args.run_id
    )