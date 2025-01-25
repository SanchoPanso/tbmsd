import os
import cv2
from ultralytics import YOLO
from yolo_patch_fusion.model.model import YOLOPatch
from yolo_patch_fusion.prediction.predict import predict_dataset

images_dir = "/mnt/c/Users/Alex/Downloads/job_219_dataset_2024_10_04_20_18_11_coco 1.0/images"
labels_dir = "/mnt/c/Users/Alex/Downloads/job_219_dataset_2024_10_04_20_18_11_coco 1.0/predicts"
model = YOLOPatch('yolov8x_aerial_base_02072024.onnx', task='detect')
# model.overrides['device'] = 'cpu'
model.names


predict_dataset(images_dir, labels_dir, model)


