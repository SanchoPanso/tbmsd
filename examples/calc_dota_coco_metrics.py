import os
import json
import cv2
import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolo_patch_fusion.evaluation.convert import yolo_to_coco
from yolo_patch_fusion.evaluation.coco import load_cocodt_from_yolo, load_cocogt_from_yolo, summarize


def main():
    classes = {
        0: 'plane',
        1: 'ship',
        2: 'storage tank',
        3: 'baseball diamond',
        4: 'tennis court',
        5: 'basketball court',
        6: 'ground track field',
        7: 'harbor',
        8: 'bridge',
        9: 'large vehicle',
        10: 'small vehicle',
        11: 'helicopter',
        12: 'roundabout',
        13: 'soccer ball field',
        14: 'swimming pool',
    }
    # cocoGt = load_cocogt_from_yolo('/mnt/c/Users/Alex/Downloads/temp_labels', classes)
    # cocoDt = load_cocodt_from_yolo('tmp_pred', classes)

    image_dir = '/mnt/d/datasets/dota/DOTAv1/images/val'
    image_sizes = {}
    for fn in tqdm.tqdm(os.listdir(image_dir)):
        img = cv2.imread(os.path.join(image_dir, fn))
        name, ext = os.path.splitext(fn)
        image_sizes[name] = (img.shape[1], img.shape[0]) 

    cocoGt = load_cocogt_from_yolo('/mnt/d/datasets/dota/DOTAv1/labels/val_detect', classes, image_sizes=image_sizes)
    cocoDt = load_cocodt_from_yolo('predictions/dotav1_1024', classes, None, image_sizes=image_sizes)

    # Инициализация COCOeval
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoEval.params.maxDets = [1, 10, 100, 1000]

    step = 340
    for i in range(0, 1020, step):
        cocoEval.params.areaRng.append([i ** 2, (i + step) ** 2])
        cocoEval.params.areaRngLbl.append(f"{i} ** 2 - {i + step} ** 2")

    # Выполняем вычисление метрик
    cocoEval.evaluate()
    cocoEval.accumulate()
    
    # cocoEval.summarize()
    summarize(cocoEval)
    show_ap_by_classes(cocoEval, classes)
    print(cocoEval.eval['recall'][0, :, 0, -1])

    for i, c in classes.items():
        precision = cocoEval.eval['precision']
        print(c)

        for j in range(3):
            ap = np.mean(precision[0, :, i, 4 + j, -1])
            print(ap)


    calib_data = {str(i): {'square_points': [], 'precision_points': []} for i in range(len(classes))}
    for i, c in classes.items():
        precision = cocoEval.eval['precision']

        calib_data[str(i)]['square_points'].append(0)
        calib_data[str(i)]['precision_points'].append(0)
        prev_ap = 0

        for j in range(3):
            ap = np.mean(precision[0, :, i, 4 + j, -1])
            if ap == -1:
                ap = prev_ap
            else:
                prev_ap = ap

            calib_data[str(i)]['square_points'].append((340 * j + 170) ** 2)
            calib_data[str(i)]['precision_points'].append(ap)

        calib_data[str(i)]['square_points'].append(1024 ** 2)
        calib_data[str(i)]['precision_points'].append(calib_data[str(i)]['precision_points'][-1])

        calib_data[str(i)]['precision_points'][0] = calib_data[str(i)]['precision_points'][1]
    
    with open('calib_size_1024.json', 'w') as f:
        json.dump(calib_data, f, indent=4)
        

def show_ap_by_classes(coco_eval: COCOeval, classes: dict):
    print('\nAverage Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ]')
        
    for i, c in classes.items():
        precision = coco_eval.eval['precision']
        class_p = precision[0, :, i, 0, -1]
        class_ap = np.mean(class_p)
        print(f'{c}{' ' * max(0, 20 - len(c))}{class_ap}')


if __name__ == '__main__':
    main()
