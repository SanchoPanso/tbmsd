import os


def segmentation_to_detection(input_dir, output_dir):
    """
    Преобразует YOLO разметку сегментации в формат детекции.
    
    :param input_dir: Директория с файлами сегментации.
    :param output_dir: Директория для сохранения детекции.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r") as infile, open(output_path, "w") as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) < 6 or (len(parts) - 2) % 2 == 0:
                        raise ValueError(f"Некорректная разметка сегментации в файле {filename}: {line}")

                    class_id = parts[0]
                    points = list(map(float, parts[1:]))

                    # Извлечение минимальных и максимальных координат из списка точек
                    x_coords = points[0::2]
                    y_coords = points[1::2]

                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)

                    # Вычисление центра и размеров рамки в YOLO формате
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min

                    # Запись результата в файл
                    outfile.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Преобразование завершено. Результаты сохранены в директорию {output_dir}.")


# Пример использования
input_segmentation_dir = "/mnt/c/Users/Alex/Downloads/DOTAv1/labels/val"
output_detection_dir = "/mnt/c/Users/Alex/Downloads/DOTAv1/labels/val_detect"
segmentation_to_detection(input_segmentation_dir, output_detection_dir)
