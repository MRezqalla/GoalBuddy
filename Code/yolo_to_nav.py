#!/usr/bin/env python
from json.tool import main
from tkinter.messagebox import NO
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from yolov8_msgs.msg import BoundingBox2D
import tf2_ros
import tf2_geometry_msgs 

class Yolo_To_Goal(Node):
    def __init__(self):
        super().__init__('yolo_to_goal')
        self._sub = self.create_subscription(
            BoundingBox2D,
            'detections',
            self.callback,
            10)
        self._sub 

        self._pub = self.create_publisher(PoseStamped(), "goal_update", 10)

    # def transform_pose(self, input_pose, from_frame, to_frame):

    #     # **Assuming /tf2 topic is being broadcasted
    #     tf_buffer = tf2_ros.Buffer()
    #     listener = tf2_ros.TransformListener(tf_buffer)

    #     pose_stamped = tf2_geometry_msgs.PoseStamped()
    #     pose_stamped.pose = input_pose
    #     pose_stamped.header.frame_id = from_frame
    #     pose_stamped.header.stamp = self.get_clock().now().to_msg()

    #     try:
    #         # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
    #         output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rclpy.Duration(1))
    #         return output_pose_stamped.pose

    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #         raise


    def callback(self, data):
        centroid_x = data.center.position.x
        # centroid_y = data.center.position.y 
        width_x = data.size.x
        # width_y = data.size.y

        orth_dist = 123984 * (width_x ** (-1.099))
        dist_from_center = (0.00225 * orth_dist + 0.34894) * (centroid_x - 320)
        self.get_logger().info("Ball at position %.2f away and %.2f to the side", orth_dist, dist_from_center)


        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "optical_frame"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = dist_from_center
        goal_pose.pose.position.y = orth_dist
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self._pub.publish(goal_pose)
        print("Goal Published")


 
def main(args=None):
    rclpy.init(args=args)
    yolo_to_goal = Yolo_To_Goal()
    rclpy.spin(yolo_to_goal)
    yolo_to_goal.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()