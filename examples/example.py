import cv2
from yolo_patch_fusion.model.wrapper import YOLOInferenceWrapper


wrapper = YOLOInferenceWrapper('yolo11n.pt')
image = cv2.imread('/home/alex/workspace/YOLOPatchFusion/images/zidane3.jpg')
results = wrapper.detect(image)
print(results.boxes)

im = results.plot()
cv2.imwrite('show.jpg', im)
