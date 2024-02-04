from lidar_lite import Lidar_Lite

lidar = Lidar_Lite()

connected = lidar.connect(1)

if connected < -1:
    print("Not connected")
else:
    print("Connected")

while True:
    distance = lidar.getDistance()
    print("Distance: " + str(distance))