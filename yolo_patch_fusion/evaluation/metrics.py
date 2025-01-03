import torch
import os
from copy import deepcopy
import numpy as np
from ultralytics.utils.metrics import box_iou
from ultralytics.engine.results import Results, Boxes


def calculate_map50(gt_results, pred_results):
    """
    Расчет метрики mAP50 на основе списков результатов ground truth и предсказаний по всему датасету.
    
    :param gt_results: List[Results], список объектов Results для ground truth
    :param pred_results: List[Results], список объектов Results для предсказаний
    :return: значение mAP50
    """
    assert len(gt_results) == len(pred_results), "Ground truth и предсказания должны иметь одинаковое количество элементов."
    
    # Список для хранения всех предсказанных и истинных значений
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    
    all_gt_boxes = []
    all_gt_labels = []
    
    # Собираем все боксы и классы для всех изображений
    for gt_result, pred_result in zip(gt_results, pred_results):
        gt_boxes, gt_labels = gt_result.boxes.xyxy.cpu().numpy(), gt_result.boxes.cls.cpu().numpy()
        pred_boxes, pred_scores, pred_labels = (
            pred_result.boxes.xyxy.cpu().numpy(),
            pred_result.boxes.conf.cpu().numpy(),
            pred_result.boxes.cls.cpu().numpy(),
        )
        
        all_gt_boxes.append(gt_boxes)
        all_gt_labels.append(gt_labels)
        
        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)
        all_pred_labels.append(pred_labels)

    # Все предсказания и ground truth для всего датасета
    all_pred_boxes = np.concatenate(all_pred_boxes)
    all_pred_scores = np.concatenate(all_pred_scores)
    all_pred_labels = np.concatenate(all_pred_labels)
    
    all_gt_boxes = np.concatenate(all_gt_boxes)
    all_gt_labels = np.concatenate(all_gt_labels)

    # Уникальные классы
    classes = np.unique(np.concatenate([all_pred_labels, all_gt_labels]))
    aps = []

    for cls in classes:
        # Фильтруем предсказания и истинные значения для текущего класса
        pred_indices = all_pred_labels == cls
        gt_indices = all_gt_labels == cls
        pred_cls_boxes = all_pred_boxes[pred_indices]
        pred_cls_scores = all_pred_scores[pred_indices]
        gt_cls_boxes = all_gt_boxes[gt_indices]

        # Сортируем предсказания по убыванию вероятности
        sorted_indices = np.argsort(-pred_cls_scores)
        pred_cls_boxes = pred_cls_boxes[sorted_indices]
        pred_cls_scores = pred_cls_scores[sorted_indices]

        # Сопоставляем предсказания и истинные значения
        tp = np.zeros(len(pred_cls_boxes))
        fp = np.zeros(len(pred_cls_boxes))
        matched = np.zeros(len(gt_cls_boxes))

        for i, pred_box in enumerate(pred_cls_boxes):
            if gt_cls_boxes.size == 0:
                fp[i] = 1
                continue

            ious = box_iou(torch.tensor(pred_box)[None], torch.tensor(gt_cls_boxes))
            ious = ious.numpy()
            max_iou_idx = np.argmax(ious)
            max_iou = ious[0, max_iou_idx]

            if max_iou >= 0.5 and not matched[max_iou_idx]:
                tp[i] = 1
                matched[max_iou_idx] = 1
            else:
                fp[i] = 1

        # Рассчитываем precision и recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (len(gt_cls_boxes) + 1e-6)

        # Рассчитываем AP (average precision) для текущего класса
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        aps.append(ap)

    return np.mean(aps) if aps else 0.0



def yolo_to_results(yolo_folder, image_width, image_height, class_mapping):
    """
    Преобразует папку с разметкой YOLO в список объектов Results из Ultralytics.

    :param yolo_folder: Путь к папке с файлами разметки YOLO.
    :param image_width: Ширина изображения в пикселях.
    :param image_height: Высота изображения в пикселях.
    :param class_mapping: Словарь {class_id: class_name}, соответствие между ID и именами классов.
    :return: Список объектов Results.
    """
    results_list = []

    for filename in sorted(os.listdir(yolo_folder)):
        if filename.endswith(".txt"):
            file_path = os.path.join(yolo_folder, filename)

            # Создаем списки для данных о разметке
            boxes = []
            confidences = []
            class_ids = []

            with open(file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # Пропуск некорректных строк

                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts[:5])
                    score = float(parts[5]) if len(parts) == 6 else None

                    # Преобразование координат и размеров
                    abs_width = bbox_width * image_width
                    abs_height = bbox_height * image_height
                    abs_x_min = (x_center * image_width) - (abs_width / 2)
                    abs_y_min = (y_center * image_height) - (abs_height / 2)
                    abs_x_max = abs_x_min + abs_width
                    abs_y_max = abs_y_min + abs_height

                    # Добавляем данные в соответствующие списки
                    score = score if score is not None else 1.0
                    boxes.append([abs_x_min, abs_y_min, abs_x_max, abs_y_max, score, int(class_id)])

            # Преобразуем данные в формат numpy
            boxes = torch.tensor(boxes, dtype=torch.float32)
            # Создаем объект Results
            results = Results(boxes=boxes, orig_img=np.zeros((image_height, image_width, 3)), path=None, names=class_mapping)
            results_list.append(results)

    return results_list



if __name__ == '__main__':
    from ultralytics import YOLO

    # Загружаем ground truth и предсказания
    model = YOLO("yolov8n.pt")

    # Генерация ground truth и предсказаний для набора изображений
    pred_results = model(
        [
            '/home/alex/workspace/YOLOPatchFusion/datasets/coco8/images/val/000000000036.jpg', 
            '/home/alex/workspace/YOLOPatchFusion/datasets/coco8/images/val/000000000042.jpg',
            '/home/alex/workspace/YOLOPatchFusion/datasets/coco8/images/val/000000000049.jpg',
            '/home/alex/workspace/YOLOPatchFusion/datasets/coco8/images/val/000000000061.jpg',
        ],
        conf=0.1,
    )
    # gt_results = deepcopy(pred_results)
    # gt_results[1].boxes = Boxes(torch.zeros((0, 6)), (640, 640))
    gt_results = yolo_to_results('datasets/coco8/labels/val', 640, 640, model.names)

    print(pred_results[0].boxes, pred_results[1].boxes)

    # Расчет mAP50
    map50 = calculate_map50(gt_results, pred_results)
    print(f"mAP50: {map50:.4f}")
