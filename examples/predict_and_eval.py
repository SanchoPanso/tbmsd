from yolo_patch_fusion.prediction.predict import predict_dataset_multi_scale
from yolo_patch_fusion.evaluation.eval_yolo import eval_yolo
from yolo_patch_fusion.model.calibrator import SquareCalibrator


def main():
    images_dir = '/mnt/c/Users/Alex/Downloads/DOTAv1/images/val'
    gt_labels_dir = '/mnt/d/datasets/dota/DOTAv1/labels/val_detect'
    dt_labels_dir = "predictions/dotav1_val_predicts"
    model_path = 'dotav1_det/weights/best.pt'

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

    calibrators = None # [
    #     SquareCalibrator('calib_size_1024.json'),
    # ]

    predict_and_eval(
        classes,
        images_dir,
        gt_labels_dir,
        dt_labels_dir,
        model_path,
        patch_sizes=(1024, 2048),
        overlap=20,
        multiscale_merging_policy='hybrid_nms_v2',
        merging_policy='no_gluing',
        calibrators=calibrators,
    )


def predict_and_eval(
        classes,
        images_dir,
        gt_labels_dir,
        dt_labels_dir,
        model_path,
        patch_sizes=(1024, 2056),
        overlap=0,
        merging_policy: str = 'no_gluing',
        multiscale_merging_policy='include_all',
        calibrators=None):
    
    predict_dataset_multi_scale(
        images_dir,
        dt_labels_dir,
        model_path,
        patch_sizes,
        overlap,
        merging_policy,
        multiscale_merging_policy,
        calibrators=calibrators,
    )

    eval_yolo(
        classes,
        gt_labels_dir,
        dt_labels_dir,
        images_dir,
    )


if __name__ == '__main__':
    main()
