import os
import cv2
from ultralytics.engine.model import Model

def predict_dataset(images_dir, labels_dir, model: Model):

    os.makedirs(labels_dir, exist_ok=True)

    for fn in os.listdir(images_dir):
        name, ext = os.path.splitext(fn)
        img = cv2.imread(os.path.join(images_dir, fn))[:, :, ::-1]
        results = model.predict(img)[0]
        results.save_txt(os.path.join(labels_dir, name + '.txt'), save_conf=True)
