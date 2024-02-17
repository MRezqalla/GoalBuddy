from ultralytics import YOLO

model_n = YOLO('../../Models/Standard/yolov8n.pt')
model_n.export(format='openvino', imgsz=320)

#model_s = YOLO('yolov8s.pt')
#model_s.export(format='edgetpu')

#model_m = YOLO('yolov8m.pt')
#model_m.export(format='edgetpu')
