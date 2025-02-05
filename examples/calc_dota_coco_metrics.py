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
    cocoDt = load_cocodt_from_yolo('dotav1_val_predicts_1024', classes, None, image_sizes=image_sizes)

    # Инициализация COCOeval
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoEval.params.maxDets = [1, 10, 100, 1000]

    # for i in range(0, 1000, 200):
    #     cocoEval.params.areaRng.append([i ** 2, (i + 200) ** 2])
    #     cocoEval.params.areaRngLbl.append(f"{i} ** 2 - {i + 200} ** 2")

    # Выполняем вычисление метрик
    cocoEval.evaluate()
    cocoEval.accumulate()
    
    # cocoEval.summarize()
    summarize(cocoEval)
    show_ap_by_classes(cocoEval, classes)

    print(cocoEval.eval['recall'][0, :, 0, -1])


def show_ap_by_classes(coco_eval: COCOeval, classes: dict):
    print('\nAverage Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ]')
        
    for i, c in classes.items():
        precision = coco_eval.eval['precision']
        class_p = precision[0, :, i, 0, -1]
        class_ap = np.mean(class_p)
        print(f'{c}{' ' * max(0, 20 - len(c))}{class_ap}')


if __name__ == '__main__':
    main()
