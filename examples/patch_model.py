import cv2
from yolo_patch_fusion.model.model import YOLOPatch
from ultralytics import YOLO


model = YOLOPatch('yolo11n.onnx', task='detect')
image = cv2.imread('/home/alex/workspace/YOLOPatchFusion/images/zidane2.jpg')
results = model.predict(image, task='detect', device='cpu')
print(results[0].boxes)

im = results[0].plot()
cv2.imwrite('show.jpg', im)
