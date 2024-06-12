#!/usr/bin/env python3

from __future__ import annotations

import tf2_ros
import rospy

from typing import List
from geometry_msgs.msg import PointStamped, Point, WrenchStamped
from ros_numpy import numpify
import numpy as np
import message_filters


class CoPVisualizer:
    def __init__(self):
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        self.frame = rospy.get_param("~frame", "odom")
        self.suport_grf_thresh = rospy.get_param("~support_grf_thresh", 10)  # 10N = 1kg
        self.smoothing_window_size = rospy.get_param("~smoothing_window_size", 10)

        self.cop_pub = rospy.Publisher("~/cop", PointStamped, queue_size=1)

        self.left_ankle_history: List[np.ndarray] = []
        self.right_ankle_history: List[np.ndarray] = []

        self.left_wrench_sub = message_filters.Subscriber("left_ankle_ft", WrenchStamped)
        self.right_wrench_sub = message_filters.Subscriber("right_ankle_ft", WrenchStamped)

        self.ts = message_filters.TimeSynchronizer([self.left_wrench_sub, self.right_wrench_sub], 1)
        self.ts.registerCallback(self.wrench_callback)

    def _compute_single_support(self, wrench: WrenchStamped, sole_link_frame: str) -> np.ndarray:
        try:
            t = self.buffer.lookup_transform(sole_link_frame, wrench.header.frame_id, rospy.Time(0))
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Exception while looking up transform for wrench: {e}")
            return

        t = numpify(t.transform)

        f = t[:3, :3] @ numpify(wrench.wrench.force)

        tau = t[:3, :3] @ numpify(wrench.wrench.torque)

        sensor_offset = t[:, 3]
        d = sensor_offset[2]

        p_x = (-tau[1] - f[0] * d) / f[2]
        p_y = (-tau[0] - f[1] * d) / f[2]

        return np.array((p_x, p_y, 0.0)), f

    def wrench_callback(self, left_wrench: WrenchStamped, right_wrench: WrenchStamped):

        p_l, f_l = self._compute_single_support(left_wrench, "left_sole_link")
        p_r, f_r = self._compute_single_support(right_wrench, "right_sole_link")

        self.left_ankle_history.append(np.concatenate((p_l, f_l)))
        self.right_ankle_history.append(np.concatenate((p_r, f_r)))

        if len(self.left_ankle_history) < self.smoothing_window_size:
            rospy.loginfo("not enough measurements available")
            return

        window_l = np.stack(self.left_ankle_history[-self.smoothing_window_size:])
        window_r = np.stack(self.right_ankle_history[-self.smoothing_window_size:])

        window_l = np.mean(window_l, axis=0)
        window_r = np.mean(window_r, axis=0)

        p_l, f_l = window_l[:3], window_l[3:]
        p_r, f_r = window_r[:3], window_r[3:]

        try:
            t_l = self.buffer.lookup_transform(self.frame, "left_sole_link", rospy.Time(0))
            t_r = self.buffer.lookup_transform(self.frame, "right_sole_link", rospy.Time(0))
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Exception while looking up transform for wrench: {e}")
            return

        t_l = numpify(t_l.transform)
        t_r = numpify(t_r.transform)

        p_l = t_l @ [*p_l, 1]
        f_l = t_l[:3, :3] @ f_l
        p_r = t_r @ [*p_r, 1]
        f_r = t_r[:3, :3] @ f_r

        support_left = f_l[2] > self.suport_grf_thresh
        support_right = f_r[2] > self.suport_grf_thresh

        if support_left and not support_right:
            p_x = p_l[0]
            p_y = p_l[1]
        elif support_right and not support_left:
            p_x = p_r[0]
            p_y = p_r[1]
        elif support_left and support_right:
            p_x = (p_r[0] * f_r[2] + p_l[0] * f_l[2]) / (f_r[2] + f_l[2])
            p_y = (p_r[1] * f_r[2] + p_l[1] * f_l[2]) / (f_r[2] + f_l[2])
        else:
            rospy.logerr(f"No Support (left grf: {f_l[2]}, right grf: {f_r[2]})")
            return
        p_z = min(t_l[2, 3], t_r[2, 3])

        cop_point = Point()
        cop_point.x = p_x
        cop_point.y = p_y
        cop_point.z = p_z

        cop_point_stamped = PointStamped()
        cop_point_stamped.header.frame_id = self.frame
        cop_point_stamped.header.stamp = rospy.Time.now()
        cop_point_stamped.point = cop_point

        self.cop_pub.publish(cop_point_stamped)


def main():
    rospy.init_node("cop_visualizer")

    node = CoPVisualizer()
    rospy.loginfo("cop_visualizer initialized")
    rospy.spin()


if __name__ == "__main__":
    main()
