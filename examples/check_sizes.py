import datasculptor as dts
import matplotlib.pyplot as plt


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

labels_dir = "/mnt/d/datasets/dota/DOTAv1-split/labels/val"
annot = dts.read_yolo(labels_dir, classes=list(classes.values()))

sizes = {v: [] for k, v in classes.items()}
for image_name in annot.images:
    image = annot.images[image_name]

    for obj in image.annotations:
        x, y, w, h = obj.bbox
        cat = classes[obj.category_id]
        sizes[cat].append(max(w, h))

for c in sizes:
    plt.figure().clear()
    plt.hist(sizes[c], 10)
    plt.title(c)
    plt.savefig(f"{c}.jpg")
