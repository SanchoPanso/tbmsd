import os
import cv2
import tqdm
import shutil
from ultralytics import YOLO
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper
from yolo_patch_fusion.model.multi_scale_model import MultiPatchInference
from yolo_patch_fusion.prediction.predict import predict_dataset


def main():
    images_dir = "/mnt/c/Users/Alex/Downloads/DOTAv1/images/val"
    labels_dir = "dotav1_val_predicts_1024"

    # images_dir = "/mnt/d/datasets/dota/DOTAv1-split/images/val"
    # labels_dir = "dotav1_val_split_predicts_1024"
    merging_policy = 'no_gluing' # 'frame_aware_gluing'
    multiscale_merging_policy = 'nms'

    model = YOLOPatchInferenceWrapper('/home/alex/workspace/YOLOPatchFusion/dotav1_det/weights/best.pt')
    inferencer = MultiPatchInference(model)

    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    os.makedirs(labels_dir, exist_ok=True)

    for fn in tqdm.tqdm(sorted(os.listdir(images_dir))):
        name, ext = os.path.splitext(fn)
        img = cv2.imread(os.path.join(images_dir, fn))
        
        results = inferencer(
            img, 
            patch_sizes=(1024, 2056), 
            overlap=0, 
            merging_policy=merging_policy, 
            multiscale_merging_policy=multiscale_merging_policy,
        )[0]
        
        results.save_txt(os.path.join(labels_dir, name + '.txt'), save_conf=True)


if __name__ == '__main__':
    main()
