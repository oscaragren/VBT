from ultralytics import YOLO
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from ultralytics.models.yolo.segment import SegmentationTrainer
import math

class CustomTrainer(SegmentationTrainer):
    def _setup_scheduler(self):
        self.lf = lambda x: self.args.lr0 + 0.5 * (self.args.lr0 - self.args.lrf) * (1 + math.cos(math.pi * x))
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lf)        


model = YOLO("yolov9c-seg.yaml") # This is the model

results = model.train(trainer=CustomTrainer, data="data.yaml", epochs=100, imgsz=640, device="mps", plots=True, lr0=0.001) # mps is for Apple M1 and M2
