import os
import cv2
from ultralytics import YOLO
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper
from yolo_patch_fusion.prediction.predict import predict_dataset

images_dir = "/mnt/c/Users/Alex/Downloads/DOTAv1/images/val"
labels_dir = "dotav1_val_predicts_1024"
model = YOLOPatchInferenceWrapper('/home/alex/workspace/YOLOPatchFusion/dotav1_det/weights/best.pt')
merging_policy = 'no_gluing'

predict_dataset(images_dir, labels_dir, model, overlap=0, merging_policy=merging_policy)


