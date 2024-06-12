#!/usr/bin/env python3

from __future__ import annotations

import urdf_parser_py.urdf as urdf
import tf2_ros
import rospy

from geometry_msgs.msg import TransformStamped, PointStamped, Point
from ros_numpy import numpify
import numpy as np

import time


class CoMVisualizer:
    def __init__(self):
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.robot: urdf.Robot = urdf.Robot.from_parameter_server()

        self.frame = rospy.get_param("~frame", "base_link")

        self.com_pub = rospy.Publisher("~/com", PointStamped, queue_size=1)
        # self.timer = rospy.Timer(rospy.Duration(secs=1), self.timer_callback)

    def timer_callback(self, _=None):
        self.compute_com()

    def compute_com(self):
        masses = []
        poses = []

        for link in self.robot.links:
            link: urdf.Link = link
            if link.inertial is None or link.inertial.mass is None:
                # no intertia information available
                continue

            try:
                t: TransformStamped = self.buffer.lookup_transform(
                    self.frame, link.name, rospy.Time(0)
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logerr(f"Exception while looking up transform for {link.name}: {e}")
                return

            origin = link.inertial.origin.xyz if link.inertial.origin is not None else [0, 0, 0]

            t: np.ndarray = numpify(t.transform)
            position = t @ np.array((*origin, 1))

            poses.append(position[:3])
            masses.append(link.inertial.mass)

        poses = np.stack(poses, axis=0)
        masses = np.array(masses)

        total_mass = np.sum(masses)
        com = np.sum(poses * masses[..., None], axis=0) / total_mass
        # I end up with around 68.6263, the weight of REEM C according to google is 80kg
        rospy.loginfo(f"Total Mass: {total_mass}")
        rospy.loginfo(f"Center of Mass: {com}")

        com_point = Point()

        com_point.x = com[0]
        com_point.y = com[1]
        com_point.z = com[2]

        com_point_stamped = PointStamped()
        com_point_stamped.header.frame_id = self.frame
        com_point_stamped.header.stamp = rospy.Time.now()
        com_point_stamped.point = com_point
        self.com_pub.publish(com_point_stamped)


def main():
    rospy.init_node("com_visualizer")

    node = CoMVisualizer()
    rospy.loginfo("com_visualizer initialized")
    while not rospy.is_shutdown():
        node.timer_callback()


if __name__ == "__main__":
    main()
