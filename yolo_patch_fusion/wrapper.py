import cv2
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes


class YOLOInferenceWrapper:
    def __init__(self, model_path: str, img_size: int = 640):
        self.model = YOLO(model_path)
        self.img_size = img_size

    def split_image(self, image: np.ndarray, overlap: int = 50):
        h, w, _ = image.shape
        step = self.img_size - overlap
        crops = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                x_end = min(x + self.img_size, w)
                y_end = min(y + self.img_size, h)
                crops.append((image[y:y_end, x:x_end], x, y))
        return crops

    def infer_on_crop(self, crop: np.ndarray):
        results = self.model(crop, verbose=False)
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy().reshape(-1, 1)
        conf = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
        final_results = np.concatenate([xyxy, conf, cls], axis=1)
        return final_results  # (xmin, ymin, xmax, ymax, confidence, class)

    def merge_detections(self, detections, iou_threshold=0.5):
        geometries = [box(*det[:4]) for det in detections]
        confidences = [det[4] for det in detections]
        classes = [det[5] for det in detections]

        # Построение индекса для быстрого объединения
        indexed_geometries = [(geom, conf, cls) for geom, conf, cls in zip(geometries, confidences, classes)]
        tree = STRtree([item[0] for item in indexed_geometries])
        
        merged_objects = []
        used = set()

        for i, (geom, conf, cls) in enumerate(indexed_geometries):
            if i in used:
                continue

            overlap_indices = tree.query(geom)
            overlap_indices = [idx for idx in overlap_indices if cls == indexed_geometries[idx][2]]
            used.update(overlap_indices)

            # Объединение объектов
            combined_geom = unary_union([indexed_geometries[idx][0] for idx in overlap_indices])

            # Вычисление общей площади и весов для областей
            total_area = combined_geom.area

            intersection_subareas = []
            for i in overlap_indices:
                for j in overlap_indices:
                    if i == j:
                        continue

                    intersection_subarea = indexed_geometries[i][0].intersection(indexed_geometries[j][0]).area
                    intersection_subareas.append(intersection_subarea)

            intersection_area = sum(intersection_subareas) / 2

            non_intersection_areas = []
            for i in overlap_indices:
                geom = indexed_geometries[i][0]
                non_intersection_area = geom.area - sum(
                    geom.intersection(indexed_geometries[j][0]).area for j in overlap_indices if j != i
                )
                non_intersection_areas.append(non_intersection_area)
            
            # Расчет уверенности с учётом взвешенной суммы областей
            non_intersect_conf = 0
            for i, idx in enumerate(overlap_indices):
                confidence = indexed_geometries[idx][1] * (non_intersection_areas[i] / total_area)
                non_intersect_conf += confidence

            intersect_conf = (1 - np.prod([1 - indexed_geometries[idx][1] for idx in overlap_indices])) \
                             * (intersection_area / total_area)
            
            weighted_conf = non_intersect_conf + intersect_conf

            # Класс выбирается как наиболее частый
            most_common_class = max(
                (indexed_geometries[idx][2] for idx in overlap_indices),
                key=lambda cls: sum(1 for idx in overlap_indices if indexed_geometries[idx][2] == cls)
            )

            merged_objects.append((combined_geom.bounds, weighted_conf, most_common_class))

        # Формирование списка итоговых объектов
        result = []
        for bounds, conf, cls in merged_objects:
            xmin, ymin, xmax, ymax = bounds
            result.append([xmin, ymin, xmax, ymax, conf, cls])

        return result


    def detect(self, image: np.ndarray, overlap: int = 50):
        crops = self.split_image(image, overlap)
        detections = []
        for crop, x_offset, y_offset in crops:
            crop_detections = self.infer_on_crop(crop)
            for det in crop_detections:
                det[:4] += [x_offset, y_offset, x_offset, y_offset]  # Adjust to global coords
                detections.append(det)

        # Получение объединенных детекций
        merged_detections = self.merge_detections(detections)

        # Преобразование в формат Results
        boxes = np.array([[det[0], det[1], det[2], det[3], det[4], det[5]] for det in merged_detections])  # xyxy + conf + cls
        result = Results(orig_img=image, path=None, names=self.model.names)
        result.boxes = Boxes(boxes, image.shape[:2])
        return result


# Пример использования:
# wrapper = YOLOInferenceWrapper('yolov8.pt')
# image = cv2.imread('image.jpg')
# results = wrapper.detect(image)
# print(results.boxes)
