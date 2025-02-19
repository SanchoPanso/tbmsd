import os
import cv2
import tqdm
import shutil
from typing import Tuple, List
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper
from yolo_patch_fusion.model.multi_scale_model import MultiPatchInference
from yolo_patch_fusion.model.calibrator import Calibrator

def main():
    images_dir = '/mnt/c/Users/Alex/Downloads/DOTAv1/images/val'
    dst_images_dir='predictions/visualization_dotav1_val_default_predict/images'
    gt_labels_dir = '/mnt/d/datasets/dota/DOTAv1/labels/val_detect'
    dt_labels_dir = "predictions/visualization_dotav1_val_default_predict/labels"
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

    predict_dataset_multi_scale(
        images_dir,
        dt_labels_dir,
        dst_images_dir,
        model_path,
        patch_sizes=(1024,),
        overlap=0,
        # multiscale_merging_policy='multiplicative_score_voting'
    )


def predict_dataset_multi_scale(
        images_dir: str,
        labels_dir: str,
        dst_images_dir: str,
        model_path: str,
        patch_sizes: Tuple[int] = (1024,),
        overlap: int = 0,
        merging_policy: str = 'no_gluing',
        multiscale_merging_policy: str = 'nms',
        calibrators: List[Calibrator] = None):

    model = YOLOPatchInferenceWrapper(model_path)
    inferencer = MultiPatchInference(model)

    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(dst_images_dir, exist_ok=True)

    for fn in tqdm.tqdm(sorted(os.listdir(images_dir))):
        name, ext = os.path.splitext(fn)
        img = cv2.imread(os.path.join(images_dir, fn))
        
        results = inferencer(
            img,
            conf=0.25,
            patch_sizes=patch_sizes, 
            overlap=overlap, 
            merging_policy=merging_policy, 
            multiscale_merging_policy=multiscale_merging_policy,
            calibrators=calibrators,
        )[0]

        annot_path = os.path.join(labels_dir, name + '.txt')
        
        if not results.boxes or len(results.boxes) == 0:
            with open(annot_path, 'w') as f:
                f.write('')
        else:
            results.save_txt(annot_path, save_conf=True)
        
        vis_img = results.plot()
        cv2.imwrite(os.path.join(dst_images_dir, f"{name}.jpg"), vis_img)


if __name__ == '__main__':
    main()
