from traceback import print_tb
from unittest import result
from ultralytics import YOLO
import cv2

pose_model = YOLO('yolov8n-pose.pt')

# results = pose_model(source=0, show=True, conf=0.4, classes=0, max_det=1,stream=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # length

while True:
    _, img = cap.read()
    result = pose_model(img, conf=0.4, classes=0, max_det=1,stream=True)
    
    for r in result:
        location = (r.keypoints[0].xy[0][0].numpy()) #Nose
        location_x = int(location[0])
        location_y  = int(location[1])

    image = cv2.circle(img, (location_x,location_y), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow('YOLO V8 Detection', image)
    print("HELLO")
    # pose_tensor = output[:, pose_model.model.names.index('pose')]

    # keypoint_data = pose_tensor[0].cpu().detach().numpy()
    print(pose_model.model.names)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# for r in results:
#     print("Nose location:")
#     print(r) #nose
