import os
import pathlib
import time

from fastai.vision.all import load_learner


# ============================================================
# COMPATIBILIDAD WINDOWS -> DOCKER / LINUX
# ============================================================

if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath

class ImageClassifier:
    """
    Representa un modelo de clasificación de imágenes.

    Cada objeto de esta clase administra:
    - nombre del modelo
    - arquitectura
    - archivo .pkl
    - métricas de validación
    - carga del modelo
    - predicción
    """

    def __init__(
        self,
        name,
        architecture,
        model_file,
        models_dir,
        validation_accuracy,
        validation_error_rate,
        correct_predictions,
        validation_samples,
    ):
        self.name = name
        self.architecture = architecture
        self.model_file = model_file
        self.models_dir = pathlib.Path(models_dir)

        self.validation_accuracy = validation_accuracy
        self.validation_error_rate = validation_error_rate
        self.correct_predictions = correct_predictions
        self.validation_samples = validation_samples

        # El modelo todavía no está cargado al crear el objeto
        self._learner = None

    @property
    def model_path(self):
        """Devuelve la ruta completa del archivo del modelo."""
        return self.models_dir / self.model_file

    @property
    def is_loaded(self):
        """Indica si el modelo ya fue cargado en memoria."""
        return self._learner is not None

    def load(self):
        """
        Carga el modelo únicamente la primera vez.
        Después reutiliza el modelo almacenado en memoria.
        """

        if self._learner is not None:
            return self._learner

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo: {self.model_path}"
            )

        print(
            f"Cargando {self.architecture} "
            f"desde {self.model_path}..."
        )

        self._learner = load_learner(
            self.model_path,
            cpu=True
        )

        print(
            f"{self.architecture} "
            f"cargado correctamente."
        )

        return self._learner

    def predict(self, image_path):
        """
        Clasifica una imagen como GATO o PERRO.
        """

        learner = self.load()

        start_time = time.perf_counter()

        pred_class, pred_idx, probabilities = learner.predict(
            image_path
        )

        inference_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        dog_probability = float(
            probabilities[0].item()
        ) * 100

        cat_probability = float(
            probabilities[1].item()
        ) * 100

        pred_value = pred_class

        if isinstance(pred_value, str):
            pred_value = (
                pred_value.strip().lower() == "true"
            )

        prediction = (
            "GATO"
            if bool(pred_value)
            else "PERRO"
        )

        confidence = float(
            probabilities[int(pred_idx)].item()
        ) * 100

        return {
            "model": self.name,
            "architecture": self.architecture,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "PERRO": round(dog_probability, 2),
                "GATO": round(cat_probability, 2),
            },
            "inference_time_ms": round(
                inference_time_ms,
                2
            ),
            "validation_accuracy": self.validation_accuracy,
            "validation_error_rate": self.validation_error_rate,
        }

    def get_status(self):
        """
        Devuelve información técnica del modelo.
        """

        return {
            "architecture": self.architecture,
            "file": self.model_file,
            "exists": self.model_path.exists(),
            "loaded": self.is_loaded,
            "validation_accuracy": self.validation_accuracy,
            "validation_error_rate": self.validation_error_rate,
            "correct_predictions": self.correct_predictions,
            "validation_samples": self.validation_samples,
        }