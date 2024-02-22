import cv2
import numpy
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  
import gpiozero as io

def output_motors(centroid_x):
    if centroid_x == -1:
        rightMotor.value =0 
        leftMotor.value =0
    elif centroid_x < 320 - (center_range/2):
        leftMotor.value = 0.3
        rightMotor.value = 0.5

        print("LEFT!!!!")
    elif centroid_x > (320 - (center_range/2)) and (centroid_x < 320 + (center_range/2)):
        leftMotor.value = 0.6
        rightMotor.value = 0.6
        print("MIDDLE!!!!") 
    elif centroid_x > 320 + (center_range/2):
        rightMotor.value = 0.3
        leftMotor.value = 0.5
        print("RIGHT!!!!")

leftMotor = io.PWMLED("GPIO12")
rightMotor = io.PWMLED("GPIO13")

model = YOLO('../Models/openvino/yolov8s_openvino_model/')
# model = YOLO('GoalBuddy/Models/openvino/yolov8n_openvino_model/')
cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # length

center_range = 300 

while True:
    success, img = cap.read()

    results = model(img, classes = [32], max_det=1, conf=0.5, imgsz=320)

    if success:
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
           # cv2.imshow('YOLO V8 Detection', img) 
           output_motors(-1)
        else: 
            image = cv2.circle(img, centroid, radius=5, color=(0, 0, 255), thickness=-1)
           # cv2.imshow('YOLO V8 Detection', image)
            output_motors(centroid_x)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()


