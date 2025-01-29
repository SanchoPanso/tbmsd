import cv2
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper


wrapper = YOLOPatchInferenceWrapper('dotav1_det/weights/best.pt')
image = cv2.imread('/mnt/c/Users/Alex/Downloads/DOTAv1/images/val/P0060.jpg')

results = wrapper(image, 1024, 0)[0]
print(results.boxes)

im = results.plot()

cv2.imshow('Image', im)
cv2.waitKey()

# cv2.imwrite('show.jpg', im)
