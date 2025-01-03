import os
import cv2
from ultralytics import YOLO
from yolo_patch_fusion.model.model import YOLOPatch
from yolo_patch_fusion.prediction.predict import predict_dataset


images_dir = 'datasets/coco8/images/val'
labels_dir = 'datasets/coco8/pred/val'
model = YOLO('yolo11n.pt')

predict_dataset(images_dir, labels_dir, model)


