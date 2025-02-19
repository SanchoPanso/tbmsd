import pytest
import cv2
import sys
import torch
import numpy as np
from typing import List, Tuple
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from ultralytics.engine.results import Results, Boxes

IMAGE_PLACEHOLDER = np.zeros((100, 100, 3), dtype=np.uint8)
NAMES = {i: str(i) for i in range(80)}


def test_same_frame():
    boxes = torch.tensor([[0, 0, 10, 10, 0.9, 0], [0, 0, 20, 20, 0.9, 0]])
    result = Results(orig_img=IMAGE_PLACEHOLDER, path=None, names=NAMES)
    result.boxes = Boxes(boxes, IMAGE_PLACEHOLDER.shape[:2])

    merged = merge_with_frame_aware_gluing([result], IMAGE_PLACEHOLDER)
    assert len(merged.boxes) == 2


def test_different_frame_same_class():
    boxes_1 = torch.tensor([[0, 0, 10, 10, 0.9, 0]])
    result_1 = Results(orig_img=IMAGE_PLACEHOLDER, path=None, names=NAMES)
    result_1.boxes = Boxes(boxes_1, IMAGE_PLACEHOLDER.shape[:2])

    boxes_2 = torch.tensor([[5, 5, 15, 20, 0.8, 0]])
    result_2 = Results(orig_img=IMAGE_PLACEHOLDER, path=None, names=NAMES)
    result_2.boxes = Boxes(boxes_2, IMAGE_PLACEHOLDER.shape[:2])

    merged = merge_with_frame_aware_gluing([result_1, result_2], IMAGE_PLACEHOLDER)
    assert len(merged.boxes) == 1
    assert pytest.approx(merged.boxes.data[0, :4]) == [0, 0, 15, 20]
    assert merged.boxes.data[0, 5] == 0


def test_different_frame_different_class():
    boxes_1 = torch.tensor([[0, 0, 10, 10, 0.9, 0]])
    result_1 = Results(orig_img=IMAGE_PLACEHOLDER, path=None, names=NAMES)
    result_1.boxes = Boxes(boxes_1, IMAGE_PLACEHOLDER.shape[:2])

    boxes_2 = torch.tensor([[5, 5, 15, 20, 0.8, 1]])
    result_2 = Results(orig_img=IMAGE_PLACEHOLDER, path=None, names=NAMES)
    result_2.boxes = Boxes(boxes_2, IMAGE_PLACEHOLDER.shape[:2])

    merged = merge_with_frame_aware_gluing([result_1, result_2], IMAGE_PLACEHOLDER)
    assert len(merged.boxes) == 2
    assert pytest.approx(merged.boxes.data) == [[0, 0, 10, 10, 0.9, 0], [5, 5, 15, 20, 0.8, 1]]


def merge_with_frame_aware_gluing( 
        crop_results: List[Results], 
        orig_image: np.ndarray
    ) -> List[Results]:
        
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
    result = Results(orig_img=orig_image, path=None, names={i: str(i) for i in range(80)})
    result.boxes = Boxes(boxes.reshape(-1, 6), orig_image.shape[:2])
    return result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
