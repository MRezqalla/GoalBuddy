from traceback import print_tb
from unittest import result
from ultralytics import YOLO
import cv2

pose_model = YOLO('yolov8n-pose.pt')
result = pose_model(source=0, conf=0.4, classes=0, max_det=1,stream=True,show = True)

for r in result:
    print(r)