import os
import cv2
import tqdm
import shutil
from typing import Tuple, List
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper
from yolo_patch_fusion.model.multi_scale_model import MultiPatchInference


def predict_dataset(
        images_dir, 
        labels_dir, 
        model: YOLOPatchInferenceWrapper, 
        img_size=1024, 
        overlap=0, 
        merging_policy: str = 'no_gluing', 
        device: str = 'cpu'):

    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    os.makedirs(labels_dir, exist_ok=True)

    for fn in tqdm.tqdm(sorted(os.listdir(images_dir))):
        name, ext = os.path.splitext(fn)
        img = cv2.imread(os.path.join(images_dir, fn))[:, :, ::-1]
        results = model(img, img_size=img_size, overlap=overlap, conf=0.01, merging_policy=merging_policy)[0]
        results.save_txt(os.path.join(labels_dir, name + '.txt'), save_conf=True)


def predict_dataset_multi_scale(
        images_dir: str,
        labels_dir: str,
        model_path: str,
        patch_sizes: Tuple[int] = (1024,),
        overlap: int = 0,
        merging_policy: str = 'no_gluing',
        multiscale_merging_policy: str = 'nms'):

    model = YOLOPatchInferenceWrapper(model_path)
    inferencer = MultiPatchInference(model)

    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    os.makedirs(labels_dir, exist_ok=True)

    for fn in tqdm.tqdm(sorted(os.listdir(images_dir))):
        name, ext = os.path.splitext(fn)
        img = cv2.imread(os.path.join(images_dir, fn))
        
        results = inferencer(
            img, 
            patch_sizes=patch_sizes, 
            overlap=overlap, 
            merging_policy=merging_policy, 
            multiscale_merging_policy=multiscale_merging_policy,
        )[0]

        annot_path = os.path.join(labels_dir, name + '.txt')
        
        if len(results.boxes) == 0:
            with open(annot_path, 'w') as f:
                f.write('')
        else:
            results.save_txt(annot_path, save_conf=True)

