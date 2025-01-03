import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolo_patch_fusion.evaluation.convert import yolo_to_coco
from yolo_patch_fusion.evaluation.coco import load_cocodt_from_yolo, load_cocogt_from_yolo


def main():
    classes = {i: str(i) for i in range(80)}
    cocoGt = load_cocogt_from_yolo('/home/alex/workspace/YOLOPatchFusion/datasets/coco8/labels/val', classes)
    cocoDt = load_cocodt_from_yolo('/home/alex/workspace/YOLOPatchFusion/datasets/coco8/pred/val', classes)

    # Инициализация COCOeval
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    # Выполняем вычисление метрик
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()




if __name__ == '__main__':
    main()
