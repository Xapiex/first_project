import pathlib

from src.models.classifier import ImageClassifier


class ModelManager:
    """
    Administra los modelos de clasificación disponibles
    en el proyecto.
    """

    def __init__(self):
        base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
        models_dir = base_dir / "models"

        self.models = {
            "resnet34": ImageClassifier(
                name="resnet34",
                architecture="ResNet34",
                model_file="clasificador.pkl",
                models_dir=models_dir,
                validation_accuracy=99.59,
                validation_error_rate=0.41,
                correct_predictions=1472,
                validation_samples=1478,
            ),

            "efficientnet_b0": ImageClassifier(
                name="efficientnet_b0",
                architecture="EfficientNet-B0",
                model_file="clasificador_efficientnet_b0_e1_img224_bs64.pkl",
                models_dir=models_dir,
                validation_accuracy=97.77,
                validation_error_rate=2.23,
                correct_predictions=1445,
                validation_samples=1478,
            ),
        }

    def get_model(self, model_name="resnet34"):
        """
        Devuelve el objeto clasificador solicitado.
        """

        if model_name not in self.models:
            raise ValueError(
                f"Modelo no válido: {model_name}. "
                f"Disponibles: {list(self.models.keys())}"
            )

        return self.models[model_name]

    def predict(self, image_path, model_name="resnet34"):
        """
        Ejecuta una predicción usando uno de los modelos.
        """

        classifier = self.get_model(model_name)

        return classifier.predict(image_path)

    def compare_models(self, image_path):
        """
        Ejecuta la misma imagen sobre todos los modelos.
        """

        results = {}

        for model_name in self.models:
            results[model_name] = self.predict(
                image_path,
                model_name=model_name
            )

        return results

    def get_models_status(self):
        """
        Devuelve el estado técnico de todos los modelos.
        """

        return {
            model_name: classifier.get_status()
            for model_name, classifier in self.models.items()
        }