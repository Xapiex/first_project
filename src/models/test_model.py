# src/models/test_model.py

import argparse
import pathlib

from fastai.vision.all import (
    load_learner,
    PILImage
)


def predict_image(
    model_path,
    image_path
):
    """
    Carga un modelo FastAI exportado y realiza
    una predicción sobre una imagen nueva.
    """

    model_path = pathlib.Path(
        model_path
    )

    image_path = pathlib.Path(
        image_path
    )


    # --------------------------------------------------------
    # VALIDACIONES
    # --------------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            f"No se encontró el modelo:\n"
            f"{model_path}"
        )


    if not image_path.exists():

        raise FileNotFoundError(
            f"No se encontró la imagen:\n"
            f"{image_path}"
        )


    # --------------------------------------------------------
    # CARGAR MODELO
    # --------------------------------------------------------

    print("\n====================================")
    print("CARGANDO MODELO")
    print("====================================")

    print(
        f"Modelo: {model_path}"
    )

    learn = load_learner(
        model_path,
        cpu=True
    )


    # --------------------------------------------------------
    # CARGAR IMAGEN
    # --------------------------------------------------------

    image = PILImage.create(
        image_path
    )


    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    prediction, prediction_index, probabilities = (
        learn.predict(image)
    )

    index = int(
        prediction_index
    )


    # --------------------------------------------------------
    # CLASES
    #
    # 0 = False = Perro
    # 1 = True  = Gato
    # --------------------------------------------------------

    labels = {
        0: "Perro",
        1: "Gato"
    }

    result = labels[
        index
    ]

    confidence = float(
        probabilities[index]
    )

    probability_dog = float(
        probabilities[0]
    )

    probability_cat = float(
        probabilities[1]
    )


    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print("\n====================================")
    print("PREDICCIÓN")
    print("====================================")

    print(
        f"Imagen: {image_path.name}"
    )

    print(
        f"Predicción: {result}"
    )

    print(
        f"Confianza: "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Probabilidad Perro: "
        f"{probability_dog * 100:.2f}%"
    )

    print(
        f"Probabilidad Gato: "
        f"{probability_cat * 100:.2f}%"
    )

    print("====================================")


    return {
        "prediction": result,
        "confidence": confidence,
        "probability_dog": probability_dog,
        "probability_cat": probability_cat
    }


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        required=True
    )

    parser.add_argument(
        "--image",
        required=True
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    predict_image(
        model_path=args.model_path,
        image_path=args.image
    )