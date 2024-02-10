from ultralytics import YOLO

model_n = YOLO('yolov8n.pt')
model_n.export(format='edgetpu')

model_s = YOLO('yolov8s.pt')
model_s.export(format='edgetpu')

model_m = YOLO('yolov8m.pt')
model_m.export(format='edgetpu')