from curses import baudrate
import cv2
import os
import numpy
from math import sqrt
from ultralytics import YOLO
import serial
from threading import Lock
from ultralytics.utils.plotting import Annotator
import time

display = False
time_in_center = 0
prev_error = 0
human_state = 0

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
    centroid_x = int((b[0] + b[2]) / 2)
    box_width = b[2] - b[0]
    orth_dist = 123984 * (box_width ** (-1.099))  # Calibrated
    
    ### Horizontal Calibration ###
    x_from_center = centroid_x - frame_x   # Left is negative 
    dist_from_center = (0.00225 * orth_dist + 0.34894) * x_from_center  # Calibrated

    return orth_dist, dist_from_center

def output_motors_human(state):
    if (state == 0):
        send_command(f"m {int(-3)} {int(3)}")
        print("LEFT!!!!")

    elif (state == 1):
        print("MIDDLE!!!!")

    elif (state == 2):
        send_command(f"m {int(3)} {int(-3)}")
        print("RIGHT!!!!")


def motor_PID(error):

    base_speed = 22
    # error = centroid_x - 320 
    Kp_right = -0.03
    Kp_left = 0.03

    P_value_left = error * Kp_left
    P_value_right = error * Kp_right

    out_right = base_speed + P_value_right
    out_left = base_speed + P_value_left


    print(out_right, out_left)

    send_command(f"m {int(out_left)} {int(out_right)}")

def h_motor_PID(error):
    base_speed = 0
    # error = centroid_x - 320 
    Kp_right = -0.03
    Kp_left = 0.03

    P_value_left = error * Kp_left
    P_value_right = error * Kp_right

    out_right = base_speed + P_value_right
    out_left = base_speed + P_value_left

    print(out_right, out_left)

    send_command(f"m {int(out_left)} {int(out_right)}")



def output_motors(state):


    if (state == 0):
        send_command(f"m {int(15)} {int(22)}")
        print("LEFT!!!!")

    elif (state == 1):
        send_command(f"m {int(25)} {int(25)}")
        print("MIDDLE!!!!")

    elif (state == 2):
        send_command(f"m {int(22)} {int(15)}")
        print("RIGHT!!!!")

mutex = Lock()
serial_port = '/dev/ttyACM0'
baud_rate = 115200
print(f"Connecting to port {serial_port} at {baud_rate}.")
conn = serial.Serial(serial_port, baud_rate, timeout=1)
print(f"Connected to {conn}")
model = YOLO('../Models/TPU/yolov8m_saved_model/yolov8m_full_integer_quant_edgetpu.tflite')

cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # length

center_x = 320
ball_held = 0
ball_shot = 0

center_range = 100
state = -1
prev_state = -1

while True:
    start_time = time.perf_counter()
    print("Looking for ball")
    print("Ball shot: ")
    print(ball_shot)
    if(ball_shot):
        break
    success, img = cap.read()
    prev_state = state 
    results = model(img, classes = [0, 32], max_det=1, conf=0.3, imgsz=320, stream=1)
    
    end_time = time.perf_counter()
    print(f"time diff: {end_time - start_time} seconds")
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

                    if b[1] == 0 or b[3] == 480:
                        print("OBJ AT EDGE!!")
                    else:
                        print(f"Final Distance: {final_dist:.2f} mm")
                    print("\n")

                    if len(boxes) == 0:                
                        image = img
                        output_motors(prev_state)

                    if (int(c) == 32 and ball_held == 0 ):
                        print("1") 
                        error = centroid_x - 320
                        motor_PID(error)

                        # if (centroid_x < 320 - (center_range/2)):
                        #     state = 0
                        # elif (centroid_x > (320 - (center_range/2)) and (centroid_x < 320 + (center_range/2))):
                        #     state = 1
                        # elif (centroid_x > 320 + (center_range/2)):
                        #     state = 2

                        if (orth_dist >= 350):
                            output_motors(state)
                            print("The ortho distance is: ")
                            print((orth_dist))
                        elif (orth_dist < 350):
                            print("Arrived")
                            ball_held = 1
                    elif int(c) == 0 and ball_held == 1:
                        # if (h_centroid_x < 320 - (center_range/2)):
                        #     human_state = 0
                        if (h_centroid_x > (320 - (center_range/2)) and (h_centroid_x < 320 + (center_range/2))):
                            human_state = 1
                        # elif (h_centroid_x > 320 + (center_range/2)):
                        #     human_state = 2   
                        # output_motors_human(human_state)
                        error = h_centroid_x - 320
                        h_motor_PID(error)

                        if human_state == 1:
                            time_in_center = time_in_center + 1
                            print(time_in_center)
                            if(time_in_center == 100):
                                ball_shot = 1
                                print("Player found and centerd")
                    elif ball_held:
                        send_command(f"m {int(5)} {int(-5)}")

                    
       
    if (cv2.waitKey(1) & 0xFF == ord(' ')) or ball_shot :
        break
        
conn.close()
cap.release()
cv2.destroyAllWindows()


