import cv2
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper
from yolo_patch_fusion.model.multi_scale_model import MultiPatchInference

# Создаем обертку YOLO
wrapper = YOLOPatchInferenceWrapper("yolo11n.pt")

# Создаем класс MultiPatchInference
multi_patch_inference = MultiPatchInference(wrapper)

# Входное изображение
image = cv2.imread("/home/alex/workspace/YOLOPatchFusion/images/zidane2.jpg")

# Инференс с разными размерами патчей
patch_sizes = [320, 640]
results = multi_patch_inference(image, patch_sizes)[0]

# Результаты
print(results.boxes)

im = results.plot()

cv2.imshow('image', im)
cv2.waitKey()

# results.save('show.jpg')