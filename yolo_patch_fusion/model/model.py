from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import ROOT, yaml_load
from yolo_patch_fusion.model.tasks import PatchDetectionModel
from yolo_patch_fusion.model.predictor import PatchDetectionPredictor
from yolo_patch_fusion.model.validator import PatchDetectionValidator


class YOLOPatch(Model):
    # TODO: update comments
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": PatchDetectionValidator,
                "predictor": PatchDetectionPredictor,
            }
        }
    
    # def _load(self, weights: str, task=None) -> None:
    #     """
    #     Loads a model from a checkpoint file or initializes it from a weights file.

    #     This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
    #     up the model, task, and related attributes based on the loaded weights.

    #     Args:
    #         weights (str): Path to the model weights file to be loaded.
    #         task (str | None): The task associated with the model. If None, it will be inferred from the model.

    #     Raises:
    #         FileNotFoundError: If the specified weights file does not exist or is inaccessible.
    #         ValueError: If the weights file format is unsupported or invalid.

    #     Examples:
    #         >>> model = Model()
    #         >>> model._load("yolo11n.pt")
    #         >>> model._load("path/to/weights.pth", task="detect")
    #     """
    #     super()._load(weights, task)
    #     new_model = PatchDetectionModel()
    #     new_model.model = self.model.model
    #     new_model.yaml = self.model.yaml
    #     new_model.names = self.model.names
    #     new_model.save = self.model.save
    #     new_model.inplace = self.model.inplace
    #     self.model = new_model