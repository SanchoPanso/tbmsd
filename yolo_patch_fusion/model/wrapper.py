import cv2
import torch
import numpy as np
from enum import Enum
from typing import List, Tuple
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes


class YOLOPatchInferenceWrapper:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def __call__(self, source: np.ndarray, img_size: int = 640, overlap: int = 50, conf: float = 0.25, merging_policy: str = 'no_gluing') -> List[Results]:
        # Split image into crops
        crops = self.split_image(source, img_size, overlap)
        
        # Detect object in each crop separately 
        crop_results = self.inference_crops(crops, conf)

        # Merge all detections into one set
        merged_results = self.merge_detections(crop_results, source, merging_policy)

        # # Transform to format `Results`
        # boxes = torch.tensor([[det[0], det[1], det[2], det[3], det[4], det[5]] for det in merged_detections])  # xyxy + conf + cls
        # result = Results(orig_img=source, path=None, names=self.model.names)
        # result.boxes = Boxes(boxes.reshape(-1, 6), source.shape[:2])

        return [merged_results]
    
    @property
    def names(self):
        return self.model.names
    
    def inference_crops(self, crops: List[Tuple[np.ndarray, float, float]], conf: float = 0.25) -> List[Results]:
        # Detect object in each crop separately 
        results = []
        for crop, x_offset, y_offset in crops:
            crop_detections = self.infer_on_crop(crop, conf)
            for det in crop_detections:
                det[:4] += [x_offset, y_offset, x_offset, y_offset]     # Adjust to global coords
                
            boxes = torch.tensor(crop_detections.reshape(-1, 1, 6))  # xyxy + conf + cls
            result = Results(orig_img=crop, path=None, names=self.model.names)
            result.boxes = Boxes(boxes.reshape(-1, 6), crop.shape[:2])
            results.append(result)
        
        return results

    
    def split_image(self, image: np.ndarray, img_size: int, overlap: int) -> List[Tuple[np.ndarray, float, float]]:
        h, w, _ = image.shape
        step = img_size - overlap
        crops = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                x_end = min(x + img_size, w)
                y_end = min(y + img_size, h)
                crops.append((image[y:y_end, x:x_end], x, y))
        return crops
    
    def infer_on_crop(self, crop: np.ndarray, conf=0.25) -> np.ndarray:
        crop = cv2.resize(crop, (max(crop.shape[1], 5), max(crop.shape[0], 5)))
        results = self.model.predict(crop, conf=conf, verbose=False)
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy().reshape(-1, 1)
        conf = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
        final_results = np.concatenate([xyxy, conf, cls], axis=1)
        return final_results  # (xmin, ymin, xmax, ymax, confidence, class)

    def merge_detections(self, crop_results: List[Results], orig_image: np.ndarray, merging_policy: 'MergingPolicy' = 'no_gluing') -> Results:
        if merging_policy == MergingPolicy.frame_agnostic_gluing:
            return self._merge_with_simple_gluing(crop_results, orig_image)
        
        if merging_policy == MergingPolicy.frame_aware_gluing:
            return self._merge_with_frame_aware_gluing(crop_results, orig_image)

        return self._merge_without_gluing(crop_results, orig_image)

    def _merge_with_simple_gluing(self, crop_results: List[Results], orig_image: np.ndarray) -> List[List[float]]:
        
        xyxy_list = [results.boxes.xyxy.cpu().numpy() for results in crop_results]
        cls_list = [results.boxes.cls.cpu().numpy().reshape(-1, 1) for results in crop_results]
        conf_list = [results.boxes.conf.cpu().numpy().reshape(-1, 1) for results in crop_results]

        geometries = np.concatenate(xyxy_list, axis=0)
        classes = np.concatenate(cls_list, axis=0)
        confidences = np.concatenate(conf_list, axis=0)
        
        # Построение индекса для быстрого объединения
        indexed_geometries = [(box(*geom), conf, cls) for geom, conf, cls in zip(geometries, confidences, classes)]
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
            area_weights = [indexed_geometries[idx][0].area for idx in overlap_indices]
            total_area = sum(area_weights)

            # Расчет уверенности как средневзвешенной
            weighted_conf = sum(
                indexed_geometries[idx][1] * area_weights[i] / total_area
                for i, idx in enumerate(overlap_indices)
            )
            # Класс выбирается как наиболее частый
            most_common_class = max(
                (indexed_geometries[idx][2] for idx in overlap_indices),
                key=lambda cls: sum(1 for idx in overlap_indices if indexed_geometries[idx][2] == cls)
            )

            merged_objects.append((combined_geom.bounds, weighted_conf, most_common_class))

        # Формирование списка итоговых объектов
        final_detections = []
        for bounds, conf, cls in merged_objects:
            xmin, ymin, xmax, ymax = bounds
            final_detections.append([xmin, ymin, xmax, ymax, conf[0], cls[0]])

        boxes = torch.tensor(np.array(final_detections))
        result = Results(orig_img=orig_image, path=None, names=self.model.names)
        result.boxes = Boxes(boxes.reshape(-1, 6), orig_image.shape[:2])
        return result

    def _merge_without_gluing(self, crop_results: List[Results], orig_image: np.ndarray) -> Results:

        xyxy_list = [results.boxes.xyxy.cpu().numpy() for results in crop_results]
        cls_list = [results.boxes.cls.cpu().numpy().reshape(-1, 1) for results in crop_results]
        conf_list = [results.boxes.conf.cpu().numpy().reshape(-1, 1) for results in crop_results]

        xyxy = np.concatenate(xyxy_list, axis=0)
        cls = np.concatenate(cls_list, axis=0)
        conf = np.concatenate(conf_list, axis=0)
        
        final_detections = np.concatenate([xyxy, conf, cls], axis=1)
        boxes = torch.tensor(final_detections)  # xyxy + conf + cls
        result = Results(orig_img=orig_image, path=None, names=self.model.names)
        result.boxes = Boxes(boxes.reshape(-1, 6), orig_image.shape[:2])
        
        return result
    
    def _merge_with_frame_aware_gluing(
            self, 
            crop_results: List[Results], 
            orig_image: np.ndarray) -> List[List[float]]:
        
        xyxy_list = [results.boxes.xyxy.cpu().numpy() for results in crop_results]
        cls_list = [results.boxes.cls.cpu().numpy().reshape(-1, 1) for results in crop_results]
        conf_list = [results.boxes.conf.cpu().numpy().reshape(-1, 1) for results in crop_results]
    
        geometries = np.concatenate(xyxy_list, axis=0)
        classes = np.concatenate(cls_list, axis=0)
        confidences = np.concatenate(conf_list, axis=0)

        frame_ids = []
        for i, results in enumerate(crop_results):
            frame_ids += [i] * len(results.boxes.xyxy.cpu().numpy())
    
        # Построение индекса для быстрого объединения
        indexed_geometries = [
            (box(*geom), conf, cls) for geom, conf, cls in zip(geometries, confidences, classes)
        ]
        tree = STRtree([item[0] for item in indexed_geometries])
        
        merged_objects = []
        used = set()

        for i, (geom, conf, cls) in enumerate(indexed_geometries):
            if i in used:
                continue

            overlap_indices = tree.query(geom)
            overlap_indices = [idx for idx in overlap_indices if cls == indexed_geometries[idx][2]]
            overlap_indices = [idx for idx in overlap_indices if frame_ids[i] != frame_ids[idx] or i == idx]

            used.update(overlap_indices)

            if len(overlap_indices) > 1:
                pass
            
            # Объединение объектов
            combined_geom = unary_union([indexed_geometries[idx][0] for idx in overlap_indices])
            area_weights = [indexed_geometries[idx][0].area for idx in overlap_indices]
            total_area = sum(area_weights)

            # Расчет уверенности как средневзвешенной
            weighted_conf = sum(
                indexed_geometries[idx][1] * area_weights[i] / total_area
                for i, idx in enumerate(overlap_indices)
            )
            # weighted_conf = min(
            #     indexed_geometries[idx][1]
            #     for i, idx in enumerate(overlap_indices)
            # )
            # Класс выбирается как наиболее частый
            most_common_class = max(
                (indexed_geometries[idx][2] for idx in overlap_indices),
                key=lambda cls: sum(1 for idx in overlap_indices if indexed_geometries[idx][2] == cls)
            )

            merged_objects.append((combined_geom.bounds, weighted_conf, most_common_class))

        # Формирование списка итоговых объектов
        final_detections = []
        for bounds, conf, cls in merged_objects:
            xmin, ymin, xmax, ymax = bounds
            final_detections.append([xmin, ymin, xmax, ymax, conf[0], cls[0]])

        boxes = torch.tensor(np.array(final_detections))
        result = Results(orig_img=orig_image, path=None, names=self.model.names)
        result.boxes = Boxes(boxes.reshape(-1, 6), orig_image.shape[:2])
        return result
    

class MergingPolicy(str, Enum):
    no_gluing = 'no_gluing'
    frame_agnostic_gluing = 'frame_agnostic_gluing'
    frame_aware_gluing = 'frame_aware_gluing'