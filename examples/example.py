import cv2
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper


wrapper = YOLOPatchInferenceWrapper('yolo11n.pt')
image = cv2.imread('/home/alex/workspace/YOLOPatchFusion/images/zidane3.jpg')

results = wrapper(image)[0]
print(results.boxes)

im = results.plot()

cv2.imshow('Image', im)
cv2.waitKey()

# cv2.imwrite('show.jpg', im)
