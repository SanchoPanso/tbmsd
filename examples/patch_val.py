from yolo_patch_fusion.model.model import YOLOPatch

# Create a new YOLO11n-OBB model from scratch
model = YOLOPatch("yolo11n.pt")

# Train the model on the DOTAv1 dataset
results = model.val(data='coco8_custom.yaml', batch=1)
