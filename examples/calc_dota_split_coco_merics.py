import os
import json
import numpy as np
import matplotlib.pyplot as plt
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

    cocoGt = load_cocogt_from_yolo('/mnt/d/datasets/dota/DOTAv1-split/labels/val', classes)
    cocoDt = load_cocodt_from_yolo('dotav1_val_split_predicts_1024', classes, cocoGt)

    # Инициализация COCOeval
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoEval.params.maxDets = [1, 10, 100, 1000]
    
    for i in range(0, 1000, 100):
        cocoEval.params.areaRng.append([i ** 2, (i + 100) ** 2])
        cocoEval.params.areaRngLbl.append(f"{i} ** 2 - {i + 100} ** 2")

    # Выполняем вычисление метрик
    cocoEval.evaluate()
    cocoEval.accumulate()
    
    # cocoEval.summarize()
    summarize(cocoEval)
    # show_ap_by_classes(cocoEval, classes)

    print(cocoEval.eval['recall'][0, :, 0, -1])


    for i, c in classes.items():
        precision = cocoEval.eval['precision']
        class_p = precision[0, :, i, 4:, -1]
        class_ap_per_size = np.mean(class_p, axis=0)
        plt.plot(class_ap_per_size)
        plt.title(c)
        plt.savefig('show.jpg')


def show_ap_by_classes(coco_eval: COCOeval, classes: dict):
    print('\nAverage Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ]')
        
    for i, c in classes.items():
        precision = coco_eval.eval['precision']
        class_p = precision[0, :, i, 0, -1]
        class_ap = np.mean(class_p)
        print(f'{c}{' ' * max(0, 20 - len(c))}{class_ap}')


if __name__ == '__main__':
    main()
