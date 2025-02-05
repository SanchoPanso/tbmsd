import cv2
import torch
import numpy as np
from enum import Enum
from typing import List, Sequence
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper
from yolo_patch_fusion.model.wrapper import MergingPolicy
from ultralytics.engine.results import Results, Boxes


class MultiPatchInference:
    def __init__(self, wrapper: YOLOPatchInferenceWrapper):
        self.wrapper = wrapper

    def __call__(
            self, 
            image: np.ndarray, 
            patch_sizes: Sequence[int] = (640,), 
            overlap: int = 50,
            calibrators: List['Calibrator'] = None,
            merging_policy: 'MergingPolicy' = 'no_gluing',
            multiscale_merging_policy: 'MultiScaleMergingPolicy' = 'include_all') -> List[Results]:

        scale_results = []

        for i, size in enumerate(patch_sizes):
            results = self.wrapper(image, size, overlap, conf=0.01, merging_policy=merging_policy)[0]
            
            if calibrators is not None:
                results = [calibrators[i](results[0])]

            scale_results.append(results)

        # Объединяем детекции
        results = self.combine_results(scale_results, image, multiscale_merging_policy)
        return [results]

    def combine_results(
            self, 
            scale_results: List[Results], 
            image: np.ndarray,
            multiscale_merging_policy: 'MultiScaleMergingPolicy' = 'include_all') -> Results:

        if multiscale_merging_policy == MultiScaleMergingPolicy.nms:
            return self._merge_nms(scale_results, image)
        
        if multiscale_merging_policy == MultiScaleMergingPolicy.soft_nms:
            return self._merge_soft_nms(scale_results, image)
        
        if multiscale_merging_policy == MultiScaleMergingPolicy.score_voting:
            return self._merge_soft_nms(scale_results, image)
        
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
        
        xyxy = all_boxes[keep_indices]
        conf = all_scores[keep_indices].reshape(-1, 1)
        cls = all_classes[keep_indices].reshape(-1, 1)
        boxes = torch.tensor(np.concatenate([xyxy, conf, cls], axis=1))

        merged_results = Results(orig_img=image, path=None, names=self.wrapper.names)
        merged_results.boxes = Boxes(boxes.reshape(-1, 6), image.shape[:2])
        
        return merged_results
    
    def _merge_soft_nms(
            self, 
            scale_results: List[Results], 
            image: np.ndarray, 
            iou_threshold: float = 0.5, 
            sigma: float = 0.5, 
            score_threshold: float = 0.001) -> Results:
        
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
            return Results(orig_img=image, path=None)  # Если нет детекций, возвращаем пустой объект
        
        all_boxes = np.vstack(all_boxes)
        all_scores = np.hstack(all_scores)
        all_classes = np.hstack(all_classes)
        model_ids = np.hstack(model_ids)
        
        keep_indices = soft_nms(all_boxes, all_scores, iou_threshold, sigma, score_threshold)
        
        xyxy = all_boxes[keep_indices]
        conf = all_scores[keep_indices].reshape(-1, 1)
        cls = all_classes[keep_indices].reshape(-1, 1)
        boxes = torch.tensor(np.concatenate([xyxy, conf, cls], axis=1))
        
        merged_results = Results(orig_img=image, path=None, names=self.wrapper.names)
        merged_results.boxes = Boxes(boxes.reshape(-1, 6), image.shape[:2])
        
        return merged_results
    
    def _merge_with_nms(
            self, 
            scale_results: List[Results], 
            image: np.ndarray, 
            iou_threshold: float = 0.5) -> Results:
        
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
        
        merged_boxes, merged_scores, merged_classes = score_voting(all_boxes, all_scores, all_classes, iou_threshold)
        
        merged_results = Results(image=image)
        merged_results.boxes = scale_results[0].boxes.__class__(
            xyxy=merged_boxes,
            conf=merged_scores,
            cls=merged_classes
        )

        boxes = torch.tensor(np.concatenate([merged_boxes, merged_scores, merged_classes], axis=1))
        merged_results = Results(orig_img=image, path=None, names=self.wrapper.names)
        merged_results.boxes = Boxes(boxes.reshape(-1, 6), image.shape[:2])
        
        return merged_results


def soft_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5, sigma: float = 0.5, score_threshold: float = 0.001):
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        suppress = []
        for j in range(1, len(indices)):
            iou = compute_iou(boxes[i], boxes[indices[j]])
            scores[indices[j]] *= np.exp(-iou ** 2 / sigma)
            if scores[indices[j]] < score_threshold:
                suppress.append(j)
        
        indices = np.delete(indices, suppress)
        indices = np.delete(indices, 0)
    
    return keep

def score_voting(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float) -> tuple:
    indices = np.argsort(scores)[::-1]
    merged_boxes = []
    merged_scores = []
    merged_classes = []
    
    while len(indices) > 0:
        i = indices[0]
        current_box = boxes[i]
        current_score = scores[i]
        current_class = classes[i]
        
        overlapping_boxes = [current_box]
        overlapping_scores = [current_score]
        
        suppress = []
        for j in range(1, len(indices)):
            iou = compute_iou(current_box, boxes[indices[j]])
            if iou > iou_threshold and classes[indices[j]] == current_class:
                overlapping_boxes.append(boxes[indices[j]])
                overlapping_scores.append(scores[indices[j]])
                suppress.append(j)
        
        # Обновляем координаты бокса как средневзвешенное по уверенности
        overlapping_boxes = np.array(overlapping_boxes)
        overlapping_scores = np.array(overlapping_scores)
        total_score = overlapping_scores.sum()
        weighted_box = np.sum(overlapping_boxes.T * overlapping_scores, axis=1) / total_score
        
        merged_boxes.append(weighted_box)
        merged_scores.append(total_score / len(overlapping_scores))  # Средняя уверенность
        merged_classes.append(current_class)
        
        indices = np.delete(indices, suppress)
        indices = np.delete(indices, 0)
    
    return np.array(merged_boxes), np.array(merged_scores), np.array(merged_classes)


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
    soft_nms = 'soft_nms'
    score_voting = 'score_voting'


class Calibrator:
    def __init__(self):
        pass

    def __call__(self, src_result: Results) -> Results:
        pass
