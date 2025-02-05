import os

def yolo_to_coco(
        yolo_folder: str, 
        class_mapping: dict, 
        image_sizes=None,
        image_width: int = 640, 
        image_height: int = 640) -> dict:
    
    """
    Преобразует разметку YOLO из файлов в папке в формат COCO.

    :param yolo_folder: Путь к папке с файлами разметки YOLO.
    :param image_width: Ширина изображения в пикселях.
    :param image_height: Высота изображения в пикселях.
    :param class_mapping: Словарь {class_id: class_name}, соответствие между ID и именами классов.
    :return: Словарь в формате COCO.
    """
    coco_annotations = []
    coco_images = []
    coco_categories = []

    annotation_id = 1
    image_id = 1

    # Создаем категории на основе class_mapping
    for class_id, class_name in class_mapping.items():
        coco_categories.append({
            "id": class_id + 1,
            "name": class_name
        })

    for filename in sorted(os.listdir(yolo_folder)):
        if filename.endswith(".txt"):
            # Создаем запись для изображения

            name, _ = os.path.splitext(filename)
            if image_sizes is not None and name in image_sizes:
                width, height = image_sizes[name]
            else:
                width, height = image_width, image_height
            image_entry = {
                "id": image_id,
                "file_name": filename.replace(".txt", ".jpg"),
                "width": width,
                "height": height
            }
            coco_images.append(image_entry)

            file_path = os.path.join(yolo_folder, filename)
            with open(file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # Пропуск некорректных строк

                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts[:5])
                    score = float(parts[5]) if len(parts) == 6 else None

                    # Преобразование координат и размеров
                    abs_width = bbox_width * width
                    abs_height = bbox_height * height
                    abs_x = (x_center * width) - (abs_width / 2)
                    abs_y = (y_center * height) - (abs_height / 2)

                    # Создаем аннотацию в формате COCO
                    coco_annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "bbox": [abs_x, abs_y, abs_width, abs_height],
                        "category_id": int(class_id) + 1,
                        "area": abs_width * abs_height,
                        "iscrowd": 0
                    }

                    if score is not None:
                        coco_annotation["score"] = score

                    coco_annotations.append(coco_annotation)
                    annotation_id += 1

            image_id += 1

    return {
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }


# Пример использования
if __name__ == "__main__":
    yolo_folder_path = "datasets/coco8/pred/val"
    width, height = 1920, 1080
    # classes = {
    #     0: "person",
    #     1: "car",
    #     2: "bicycle"
    # }
    classes = {i: str(i) for i in range(80)}
    coco_result = yolo_to_coco(yolo_folder_path, classes, width, height)
    print(coco_result)
