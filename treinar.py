from ultralytics import YOLO

# Carregar modelo base
model = YOLO("yolov8n.pt")

# Treinar
model.train(
    data="C:/EMPRENDE/LifeCare/dataset/data.yaml",
    epochs=20,
    imgsz=640
)