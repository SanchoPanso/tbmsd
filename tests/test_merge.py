import sys
import pytest
from shapely.geometry import box
from yolo_patch_fusion.wrapper import YOLOInferenceWrapper  # Замените на реальный модуль, где находится ваш класс


@pytest.fixture
def yolo_wrapper():
    # Создаем объект обертки YOLO
    return YOLOInferenceWrapper("yolo11n.pt")


def test_merge_detections_no_overlap(yolo_wrapper: YOLOInferenceWrapper):
    detections = [
        [0, 0, 10, 10, 0.9, 1],
        [20, 20, 30, 30, 0.8, 2],
    ]
    result = yolo_wrapper.merge_detections(detections)
    assert len(result) == 2
    assert result[0][4] == 0.9
    assert result[1][4] == 0.8


def test_merge_detections_with_overlap(yolo_wrapper: YOLOInferenceWrapper):
    detections = [
        [0, 0, 10, 10, 0.9, 1],
        [5, 5, 15, 15, 0.8, 1],
    ]
    result = yolo_wrapper.merge_detections(detections)
    assert len(result) == 1  # Объекты должны объединиться
    merged_bounds = result[0][:4]
    assert merged_bounds[0] <= 0  # xmin
    assert merged_bounds[1] <= 0  # ymin
    assert merged_bounds[2] >= 15  # xmax
    assert merged_bounds[3] >= 15  # ymax
    assert abs(result[0][4] - 0.85) < 1e-3  # Проверка уверенности


def test_merge_detections_different_classes(yolo_wrapper: YOLOInferenceWrapper):
    detections = [
        [0, 0, 10, 10, 0.9, 1],
        [5, 5, 15, 15, 0.8, 2],
    ]
    result = yolo_wrapper.merge_detections(detections)
    assert len(result) == 2  # Объекты не должны объединяться из-за разных классов


def test_merge_detections_same_class_weighted_confidence(yolo_wrapper: YOLOInferenceWrapper):
    detections = [
        [0, 0, 10, 10, 0.9, 1],  # Площадь = 100
        [5, 5, 15, 15, 0.8, 1],  # Площадь = 100
    ]
    result = yolo_wrapper.merge_detections(detections)
    assert len(result) == 1  # Объекты должны объединиться
    merged_confidence = result[0][4]
    assert abs(merged_confidence - 0.85) < 1e-3  # Средневзвешенная уверенность


def test_merge_detections_empty_input(yolo_wrapper: YOLOInferenceWrapper):
    detections = []
    result = yolo_wrapper.merge_detections(detections)
    assert result == []  # Для пустого ввода результат тоже должен быть пустым


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
