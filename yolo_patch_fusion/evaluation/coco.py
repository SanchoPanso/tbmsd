import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolo_patch_fusion.evaluation.convert import yolo_to_coco


def load_cocogt_from_yolo(yolo_folder: str, 
        class_mapping: dict, 
        image_width: int = 640, 
        image_height: int = 640) -> COCO:
    
    gt_data = yolo_to_coco(yolo_folder, class_mapping, image_width, image_height)

    for ann in gt_data["annotations"]:
        if 'score' in ann:
            ann.pop('score')

    coco_gt = COCO()
    coco_gt.dataset = gt_data    # Загружаем данные из словаря
    coco_gt.createIndex()        # Создаём индексы для быстрого доступа

    return coco_gt


def load_cocodt_from_yolo(yolo_folder: str, 
        class_mapping: dict, 
        image_width: int = 640, 
        image_height: int = 640) -> COCO:
    
    dt_data = yolo_to_coco(yolo_folder, class_mapping, image_width, image_height)
    
    for ann in dt_data["annotations"]:
        if 'score' not in ann:
            ann['score'] = 1

    coco_dt = COCO()
    coco_dt.dataset = dt_data    # Загружаем данные из словаря
    coco_dt.createIndex()        # Создаём индексы для быстрого доступа

    return coco_dt


def main():
    
    gt_coco_path = 'datasets/3764_3766_gt.json'
    pred_coco_path = 'datasets/3764_3766_pred_1280.json'

    with open(gt_coco_path) as f:
        gt_data = json.load(f)

    with open(pred_coco_path) as f:
        pred_data = json.load(f)

    cocoGt = COCO()
    cocoGt.dataset = gt_data    # Загружаем данные из словаря
    cocoGt.createIndex()        # Создаём индексы для быстрого доступа

    # Загружаем предсказания из словаря (как результат)
    cocoDt = cocoGt.loadRes(pred_data['annotations'])

    # Инициализация COCOeval
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoEval.params.maxDets = [1, 10, 100, 1000]

    area_ranges = {
        'all': [0 ** 2, 1e5 ** 2], 
    }
    step = 50
    for i in range(0, 500, step):
        area_ranges[str(i)] = [i ** 2, (i + step) ** 2]

    cocoEval.params.areaRng = list(area_ranges.values())
    cocoEval.params.areaRngLbl = list(area_ranges.keys())

    # Выполняем вычисление метрик
    cocoEval.evaluate()
    cocoEval.accumulate()
    summarize(cocoEval)


def summarize(coco_eval: COCOeval):
    params = coco_eval.params
    _summarize(coco_eval, 1, maxDets=coco_eval.params.maxDets[-1])
    _summarize(coco_eval, 1, iouThr=.5, maxDets=coco_eval.params.maxDets[-1])
    _summarize(coco_eval, 1, iouThr=.75, maxDets=coco_eval.params.maxDets[-1])


    for ap in [0, 1]:
        for label in params.areaRngLbl:
            _summarize(coco_eval, ap, iouThr=.5, areaRng=label, maxDets=coco_eval.params.maxDets[-1])
        

        for i in range(len(params.maxDets)):
            _summarize(coco_eval, ap, iouThr=.5, maxDets=coco_eval.params.maxDets[i])


def _summarize(coco_eval, ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        eval = coco_eval.eval

        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s


if __name__ == '__main__':
    main()
