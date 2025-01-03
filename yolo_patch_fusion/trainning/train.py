from ultralytics import YOLO

# Create a new YOLO11n-OBB model from scratch
model = YOLO("yolo11n.pt")

# Train the model on the DOTAv1 dataset
results = model.train(data="DOTAv1.yaml", epochs=5, imgsz=1024, batch=1)
