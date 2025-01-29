import os
import cv2
import tqdm
import shutil
from yolo_patch_fusion.model.wrapper import YOLOPatchInferenceWrapper

def predict_dataset(images_dir, labels_dir, model: YOLOPatchInferenceWrapper, merging_policy: str, device: str = 'cpu'):

    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    os.makedirs(labels_dir, exist_ok=True)

    for fn in tqdm.tqdm(sorted(os.listdir(images_dir))):
        name, ext = os.path.splitext(fn)
        img = cv2.imread(os.path.join(images_dir, fn))[:, :, ::-1]
        results = model(img, img_size=1024, overlap=0, merging_policy=merging_policy)[0]
        results.save_txt(os.path.join(labels_dir, name + '.txt'), save_conf=True)
