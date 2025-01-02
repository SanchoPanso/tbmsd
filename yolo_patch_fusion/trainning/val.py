from ultralytics import YOLO

# Create a new YOLO11n-OBB model from scratch
model = YOLO("yolo11n-obb.pt")

# Train the model on the DOTAv1 dataset
results = model.val(data="DOTAv1.yaml", imgsz=1024, batch=1)
