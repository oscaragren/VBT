from ultralytics import YOLO


model = YOLO("yolov8n-seg.pt")

results = model("../data/test/images/")

for i, res in enumerate(results):
    res.save(filename=f"./predictions/result_{i}.jpg")