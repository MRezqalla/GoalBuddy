import cv2
import numpy
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  
import gpiozero as io

def print_relative_location(centroid_x):
    if centroid_x < 320 - (center_range/2):
        print("LEFT!!!!")
    elif centroid_x > (320 - (center_range/2)) and (centroid_x < 320 + (center_range/2)):
        print("MIDDLE!!!!") 
    elif centroid_x > 320 + (center_range/2):
        print("RIGHT!!!!")

model = YOLO('../Models/openvino/yolov8n_openvino_model/')
cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # length

leftM = io.Motor(11,13)
leftM.forward(1)

center_range = 150 

while True:
    _, img = cap.read()

    results = model(img, classes = [0,32], max_det=1, conf=0.5, imgsz=640)

    for r in results:
        annotator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            print(b)
            c = box.cls
            d = box.conf.item()
            centroid_x = int((b[0] + b[2]) / 2)
            centroid_y = int((b[1] + b[3]) / 2)
            centroid = (int(centroid_x),int(centroid_y))
            annotator.box_label(b, r.names[int(c)] + ", conf = " + str(round(d,3)))
          
    img = annotator.result()  
    if len(boxes) == 0:
      cv2.imshow('YOLO V8 Detection', img) 
    else: 
        image = cv2.circle(img, centroid, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imshow('YOLO V8 Detection', image)
        print_relative_location(centroid_x)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()


