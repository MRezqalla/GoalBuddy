from curses import baudrate
import cv2
import numpy
from math import sqrt
from ultralytics import YOLO
import serial
from threading import Lock
from ultralytics.utils.plotting import Annotator
#import gpiozero as io

def send_command(cmd_string):
    
    mutex.acquire()
    try:
        cmd_string += "\r"
        conn.write(cmd_string.encode("utf-8"))

        ## Adapted from original
        c = ''
        value = ''
        while c != '\r':
            c = conn.read(1).decode("utf-8")
            if (c == ''):
                print("Error: Serial timeout on command: " + cmd_string)
                return ''
            value += c

        value = value.strip('\r')
        return value
    finally:
        mutex.release()


def get_dist(b):
    frame_x = 320
    # frame_y = 240
    centroid_x = int((b[0] + b[2]) / 2)
    # centroid_y = int((b[1] + b[3]) / 2)
    box_width = b[2] - b[0]
    #print(f"box_height: {box_height}\n\n")

    # For calibration
    #print(f"Height of Box: {box_height:.2f}\n\n")

    orth_dist = 123984 * (box_width ** (-1.099))  # Calibrated
    
    ### Horizontal Calibration ###
    x_from_center = centroid_x - frame_x   # Left is negative 
    #print(f"From center (px): {x_from_center:.2f}\n\n")
    dist_from_center = (0.00225 * orth_dist + 0.34894) * x_from_center  # Calibrated

    return orth_dist, dist_from_center


def output_motors(state):
    if (state == 0):
        send_command(f"m {int(10)} {int(15)}")
        #leftMotor.value = 0.3
        #rightMotor.value = 0.5
        print("LEFT!!!!")

    elif (state == 1):
        send_command(f"m {int(10)} {int(10)}")
        #leftMotor.value = 0.5
        #rightMotor.value = 0.5
        print("MIDDLE!!!!")

    elif (state == 2):
        send_command(f"m {int(15)} {int(10)}")
        #rightMotor.value = 0.3
        #leftMotor.value = 0.5
        print("RIGHT!!!!")


def get_coeffs(x1, y1, x2, y2):
    m =  (y2 - y1) / (x2 - x1)
    n = y1 - (m * x1)
    return m, n

# def get_dist(b): # box.xyxy
#     # distance from camera to object measured (centimeter) 
#     Known_distance = 1500  # Arbitrary value
#     # ball size 
#     Known_width = 228  # Arbitrary value
#     in_frame_width = 76
    
#     focal_length = (in_frame_width* Known_distance)/ Known_width
#     distance = (Known_width * focal_length)/(b[2] - b[0])

#     return distance

#leftMotor = io.PWMLED("GPIO12")
#rightMotor = io.PWMLED("GPIO13")

mutex = Lock()
serial_port = '/dev/tty/USB0'
baud_rate = 57600
print(f"Connecting to port {serial_port} at {baud_rate}.")
conn = serial.Serial(serial_port, baud_rate, timeout=1.0)
print(f"Connected to {conn}")


model = YOLO('../Models/Standard/yolov8s.pt')
# model = YOLO('GoalBuddy/Models/openvino/yolov8n_openvino_model/')
cap = cv2.VideoCapture(1)
cap.set(3, 640) # width
cap.set(4, 480) # length

### Depth Calibration ###
# h = Height of bounding box (px) and d = distance irl (mm)
# Record at 2 different locations (can be fine-tuned if relationship is non-linear)
h1, d1 = 100, 800
h2, d2 = 69, 1200
#3900 d, 28 h
#3400 d, 35 h
#2900 d, 42 h
#2400 d, 52 h
#1900 d, 73 h
#1400 d, 131 h  
# Coefficients to find Orthogonal distance to camera regardless of height (distance) = m*(bounding box height) + n
m, n = get_coeffs(h1, d1, h2, d2)
frame_x, frame_y = 320, 240  # middle of frame

center_range = 300 
state = -1
prev_state = -1
while True:
    success, img = cap.read()
    prev_state = state 
    results = model(img, classes = [32, 67], max_det=1, conf=0.3, imgsz=320, stream=1)

    if 1:
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

                box_height = b[2] - b[0]
                #print(f"Width: {box_height}\n\n")

                #orth_dist = m * box_height + n  # Orthogonal distance to camera regardless of how high the obj is
                orth_dist = 123984 * (box_height ** (-1.099))
                x_from_center = abs(centroid_x - frame_x)
                #y_from_center = centroid_y - frame_y

                final_dist = sqrt(orth_dist ** 2 + x_from_center ** 2)
                print(f"Y AXIS: {centroid_y}")
                final_dist, x_dist = get_dist(b)
                #final_dist = get_dist(b)
                ### FOR DEPTH CALIBRATION ###
                #print(f"Bounding box height: {box_height}")
                ### ###

                if b[1] == 0 or b[3] == 480:
                    print("OBJ AT EDGE!!")
                else:
                    print(f"Final Distance: {final_dist:.2f} mm")
                print("\n")

                if len(boxes) == 0:
                    # cv2.imshow('YOLO V8 Detection', img) 
                    image = img
                    output_motors(prev_state)
                    #print("here")

                else:    
                    #print("there")
                    image = cv2.circle(img, centroid, radius=5, color=(0, 0, 255), thickness=-1)
                
                cv2.imshow('YOLO V8 Detection', image)
                if (centroid_x < 320 - (center_range/2)):
                    state = 0
                elif (centroid_x > (320 - (center_range/2)) and (centroid_x < 320 + (center_range/2))):
                    state = 1
                elif (centroid_x > 320 + (center_range/2)):
                    state = 2
                #print(state)

                #output_motors(state)

                annotator.box_label(b, r.names[int(c)] + ", conf = " + str(round(d,3)))
          
        img = annotator.result()  

       
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
        
conn.close()
cap.release()
cv2.destroyAllWindows()


