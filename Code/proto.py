from curses import baudrate
import cv2
import os
import numpy
from math import sqrt
from ultralytics import YOLO
import serial
from threading import Lock
from ultralytics.utils.plotting import Annotator

display = False
time_in_center = 0

# def find_player():
#     human_state = 0
#     while ball_shot == 0:
#         print("Looking for player")
#         success, img = cap.read()
#         results = model(img, classes = 0, max_det=1, conf=0.4, imgsz=320, stream=1)
#         print(human_state)
#         for r in results:
#             print("gg")
#             boxes = r.boxes
#             for box in boxes:
#                 b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
#                 c = box.cls
#                 centroid_x = int((b[0] + b[2]) / 2)
            

#         if (centroid_x < 320 - (center_range/2)):
#             human_state = 0
#         elif (centroid_x > (320 - (center_range/2)) and (centroid_x < 320 + (center_range/2))):
#             human_state = 1
#         elif (centroid_x > 320 + (center_range/2)):
#             human_state = 2   

#         if human_state == 1:
#             ball_shot == 1
#             print("Player found and centerd")
#         else:
#             output_motors_human(human_state)

def send_command(cmd_string):
    
    mutex.acquire()
    try:
        cmd_string += "\r"
        conn.write(cmd_string.encode("utf-8"))
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

def output_motors_human(state):
    if (state == 0):
        send_command(f"m {int(-3)} {int(3)}")
        #leftMotor.value = 0.3
        #rightMotor.value = 0.5
        print("LEFT!!!!")

    elif (state == 1):
        print("MIDDLE!!!!")

    elif (state == 2):
        send_command(f"m {int(3)} {int(-3)}")
        #rightMotor.value = 0.3
        #leftMotor.value = 0.5
        print("RIGHT!!!!")


def output_motors(state):
    if (state == 0):
        send_command(f"m {int(15)} {int(22)}")
        #leftMotor.value = 0.3
        #rightMotor.value = 0.5
        print("LEFT!!!!")

    elif (state == 1):
        send_command(f"m {int(25)} {int(25)}")
        #leftMotor.value = 0.5
        #rightMotor.value = 0.5
        print("MIDDLE!!!!")

    elif (state == 2):
        send_command(f"m {int(22)} {int(15)}")
        #rightMotor.value = 0.3
        #leftMotor.value = 0.5
        print("RIGHT!!!!")


def get_coeffs(x1, y1, x2, y2):
    m =  (y2 - y1) / (x2 - x1)
    n = y1 - (m * x1)
    return m, n

mutex = Lock()
serial_port = '/dev/ttyACM0'
baud_rate = 57600
print(f"Connecting to port {serial_port} at {baud_rate}.")
conn = serial.Serial(serial_port, baud_rate, timeout=1)
print(f"Connected to {conn}")

# model = YOLO('../Models/Standard/yolov8s.pt')
model = YOLO('../Models/TPU/yolov8m_saved_model/yolov8m_full_integer_quant_edgetpu.tflite')
cap = cv2.VideoCapture(0)
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
ball_held = 0
ball_shot = 0

center_range = 200 
state = -1
prev_state = -1
while True:
    print("Looking for ball")
    print("Ball shot: ")
    print(ball_shot)
    if(ball_shot):
        break
    success, img = cap.read()
    prev_state = state 
    results = model(img, classes = [0, 32], max_det=1, conf=0.3, imgsz=320, stream=1)

    if success:
        for r in results:
            if(ball_shot):
                break
            annotator = Annotator(img)
            boxes = r.boxes
            if (len(boxes) == 0 and ball_held == 1):
                send_command(f"m {int(5)} {int(-5)}")
            for box in boxes:
                if ball_held == 0 or ball_held == 1:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    print(b)
                    c = box.cls
                    d = box.conf.item()
                    
                    centroid_x = int((b[0] + b[2]) / 2)
                    centroid_y = int((b[1] + b[3]) / 2)
                    if int(c) == 0:
                        h_centroid_x = int((b[0] + b[2]) / 2)
                        h_centroid_y = int((b[1] + b[3]) / 2)

                    orth_dist, x_from_center = get_dist(b)
                    orth_dist = float(orth_dist)
                    x_from_center = float(x_from_center)
                    

                    final_dist = sqrt(orth_dist ** 2 + x_from_center ** 2)
                    # print(f"Y AXIS: {centroid_y}")
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
                        image = img
                        output_motors(prev_state)

                    # else:    
                    #     image = cv2.circle(img, centroid, radius=5, color=(0, 0, 255), thickness=-1)
                    
                    if display:
                        cv2.imshow('YOLO V8 Detection', image)

                    if (int(c) == 32 and ball_held == 0 ):
                        print("1")

                        if (centroid_x < 320 - (center_range/2)):
                            state = 0
                        elif (centroid_x > (320 - (center_range/2)) and (centroid_x < 320 + (center_range/2))):
                            state = 1
                        elif (centroid_x > 320 + (center_range/2)):
                            state = 2

                        if (orth_dist >= 600):
                            output_motors(state)
                            print("The ortho distance is: ")
                            print((orth_dist))
                        elif (orth_dist < 600):
                            print("Arrived")
                            ball_held = 1
                    elif int(c) == 0 and ball_held == 1:
                        print("2")
                        print("HUMAN")
                        print(h_centroid_x)
                        # print(human_state)
                        if (h_centroid_x < 320 - (center_range/2)):
                            human_state = 0
                        elif (h_centroid_x > (320 - (center_range/2)) and (h_centroid_x < 320 + (center_range/2))):
                            human_state = 1
                        elif (h_centroid_x > 320 + (center_range/2)):
                            human_state = 2   
                        output_motors_human(human_state)

                        if human_state == 1:
                            time_in_center = time_in_center + 1
                            print(time_in_center)
                            if(time_in_center == 100):
                                ball_shot = 1
                                print("Player found and centerd")
                    elif ball_held:
                        print("3")
                        send_command(f"m {int(5)} {int(-5)}")
                        print("HELLO?")
                    print("MAW")
                    print(ball_held)

                            
                    # else:
                    #     output_motors_human(human_state)
                        # break
                    

                annotator.box_label(b, r.names[int(c)] + ", conf = " + str(round(d,3)))
            
            # if (ball_held == 1 and ball_shot == 0 and ):
            #     output_motors_human(0)
            #     # break


          
        img = annotator.result()  

       
    if (cv2.waitKey(1) & 0xFF == ord(' ')) or ball_shot :
        break
        
conn.close()
cap.release()
cv2.destroyAllWindows()


