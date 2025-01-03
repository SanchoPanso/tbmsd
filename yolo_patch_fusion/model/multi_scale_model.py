import cv2
import torch
import numpy as np
from yolo_patch_fusion.model.model import YOLOPatch
from ultralytics.engine.results import Results, Boxes


class MultiPatchInference:
    def __init__(self, wrapper: YOLOPatch):
        self.wrapper = wrapper

    def infer_with_multiple_patches(self, image: np.ndarray, patch_sizes: list, overlap: int = 50):
        """
        Выполняет инференс изображения с разными размерами патчей.
        
        :param image: Входное изображение.
        :param patch_sizes: Список размеров патчей.
        :param overlap: Перекрытие между патчами.
        :return: Итоговый объект Results с объединенными детекциями.
        """
        all_detections = []

        for size in patch_sizes:
            # Устанавливаем размер патча в обертке
            self.wrapper.img_size = size
            # Выполняем инференс для текущего размера патча
            results = self.wrapper.predict(image)
            # Извлекаем детекции из результатов
            for box in results[0].boxes.data:
                all_detections.append(box.cpu().numpy())

        # Объединяем детекции
        return self.combine_results(all_detections, image)

    def combine_results(self, detections: list, image: np.ndarray, iou_threshold: float = 0.5):
        """
        Объединяет результаты инференса, устраняя пересечения с высоким IoU.
        
        :param detections: Список всех детекций с разных размеров патчей.
        :param image: Входное изображение.
        :param iou_threshold: Порог IoU для объединения детекций.
        :return: Итоговый объект Results с объединенными детекциями.
        """
        # Преобразование детекций в массив numpy
        detections = np.array(detections)

        # Если нет детекций, возвращаем пустой результат
        if len(detections) == 0:
            return Results(orig_img=image, path=None, names=self.wrapper.model.names)

        # Сортировка по уверенности
        detections = detections[detections[:, 4].argsort()[::-1]]  # Сортировка по убыванию confidence

        merged_detections = []

        while len(detections) > 0:
            # Берем детекцию с максимальной уверенностью
            best_det = detections[0]
            merged_detections.append(best_det)

            # Рассчитываем IoU с другими детекциями
            ious = self._calculate_iou(best_det[:4], detections[:, :4])

            # Оставляем только те, что имеют IoU меньше порога
            detections = detections[ious < iou_threshold]

        # Преобразуем в формат Results
        boxes = torch.tensor([[det[0], det[1], det[2], det[3], det[4], det[5]] for det in merged_detections])  # xyxy + conf + cls
        
        final_result = Results(orig_img=image, path=None, names=self.wrapper.names)
        final_result.boxes = Boxes(boxes, image.shape[:2])

        return [final_result]

    @staticmethod
    def _calculate_iou(box: np.ndarray, boxes: np.ndarray):
        """
        Рассчитывает IoU между одной рамкой и набором рамок.
        
        :param box: Одна рамка [xmin, ymin, xmax, ymax].
        :param boxes: Набор рамок [[xmin, ymin, xmax, ymax], ...].
        :return: Вектор IoU для каждой рамки.
        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        # Вычисление площади пересечения
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Вычисление площади рамок
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Вычисление IoU
        union = box_area + boxes_area - intersection
        iou = intersection / union
        return iou


if __name__ == '__main__':
    # Создаем обертку YOLO
    wrapper = YOLOPatch("yolo11n.pt")

    # Создаем класс MultiPatchInference
    multi_patch_inference = MultiPatchInference(wrapper)

    # Входное изображение
    image = cv2.imread("/home/alex/workspace/YOLOPatchFusion/images/zidane2.jpg")

    # Инференс с разными размерами патчей
    patch_sizes = [320, 640, 1024]
    results = multi_patch_inference.infer_with_multiple_patches(image, patch_sizes, overlap=50)[0]

    # Результаты
    print(results.boxes)

    results.save('show.jpg')

