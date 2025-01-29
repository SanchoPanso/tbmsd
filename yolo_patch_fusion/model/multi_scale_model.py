import cv2
import torch
import numpy as np
from enum import Enum
from typing import List, Sequence
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper
from ultralytics.engine.results import Results, Boxes


class MultiPatchInference:
    def __init__(self, wrapper: YOLOPatchInferenceWrapper):
        self.wrapper = wrapper

    def __call__(
            self, 
            image: np.ndarray, 
            patch_sizes: Sequence[int] = (640,), 
            overlap: int = 50,
            multiscale_merging_policy: 'MultiScaleMergingPolicy' = 'include_all') -> List[Results]:
        """
        Выполняет инференс изображения с разными размерами патчей.
        
        :param image: Входное изображение.
        :param patch_sizes: Список размеров патчей.
        :param overlap: Перекрытие между патчами.
        :return: Итоговый объект Results с объединенными детекциями.
        """
        scale_results = []

        for size in patch_sizes:
            # Выполняем инференс для текущего размера патча
            results = self.wrapper(image, size, overlap)[0]
            scale_results.append(results)

        # Объединяем детекции
        results = self.combine_results(scale_results, image)
        return [results]

    def combine_results(
            self, 
            scale_results: List[Results], 
            image: np.ndarray,
            multiscale_merging_policy: 'MultiScaleMergingPolicy' = 'include_all') -> Results:

        if multiscale_merging_policy == MultiScaleMergingPolicy.nms:
            return self._merge_nms(scale_results, image)

        return self._merge_include_all(scale_results, image)

    def _merge_include_all(self, scale_results: List[Results], image: np.ndarray) -> Results:
        detections = []
        for result in scale_results:
            for box in result.boxes.data:
                detections.append(box.cpu().numpy())
        detections = np.array(detections)
        boxes = torch.tensor(detections)
        result = Results(orig_img=image, path=None, names=self.wrapper.names)
        result.boxes = Boxes(boxes.reshape(-1, 6), image.shape[:2])
        
        return result

    def _merge_nms(self, scale_results: List[Results], image: np.ndarray, iou_threshold: float = 0.5) -> Results:
        all_boxes = []
        all_scores = []
        all_classes = []
        model_ids = []  # Идентификаторы моделей для каждой детекции
        
        for model_id, result in enumerate(scale_results):
            if result.boxes is not None:
                boxes = result.boxes.xyxy  # Координаты боксов (x1, y1, x2, y2)
                scores = result.boxes.conf  # Уверенность модели в предсказании
                classes = result.boxes.cls  # Классы объектов
                
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)
                model_ids.append(np.full(len(boxes), model_id))
        
        if not all_boxes:
            return Results(image=image)  # Если нет детекций, возвращаем пустой объект
        
        all_boxes = np.vstack(all_boxes)
        all_scores = np.hstack(all_scores)
        all_classes = np.hstack(all_classes)
        model_ids = np.hstack(model_ids)
        
        keep_indices = []
        suppressed = set()
        
        for i in range(len(all_boxes)):
            if i in suppressed:
                continue
            keep_indices.append(i)
            for j in range(i + 1, len(all_boxes)):
                if model_ids[i] != model_ids[j]:  # Игнорируем пересечения из одной модели
                    iou = compute_iou(all_boxes[i], all_boxes[j])
                    if iou > iou_threshold:
                        suppressed.add(j)
        
        merged_results = Results(image=image)
        merged_results.boxes = scale_results[0].boxes.__class__(
            xyxy=all_boxes[keep_indices],
            conf=all_scores[keep_indices],
            cls=all_classes[keep_indices]
        )
        
        return merged_results


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


class MultiScaleMergingPolicy(str, Enum):
    include_all = 'include_all'
    nms = 'nms'
