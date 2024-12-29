import cv2
from yolo_patch_fusion.wrapper import YOLOInferenceWrapper


wrapper = YOLOInferenceWrapper('yolov8.pt')
image = cv2.imread('image.jpg')
results = wrapper.detect(image)
print(results.boxes)
