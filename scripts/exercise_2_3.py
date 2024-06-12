#!/usr/bin/env python3

from __future__ import annotations

import urdf_parser_py.urdf as urdf
import tf2_ros
import rospy
from visualization_msgs.msg import Marker
from typing import Dict, List, Deque

from geometry_msgs.msg import TransformStamped, Pose, PointStamped, Point, WrenchStamped, Vector3
import tf2_geometry_msgs
from ros_numpy import numpify, msgify
import numpy as np
import message_filters
from dataclasses import dataclass, field
from collections import deque
from sensor_msgs.msg import JointState, Imu

import time


G = 9.81


@dataclass
class Link:
    # name of link
    name: str
    # children
    children: List[Link] = field(default_factory=lambda: [])
    # parent
    mother: Link | None = None
    # position in world coordinates
    p: np.ndarray = np.zeros((3,))
    # attitute/rotation in world coordinates
    R: np.ndarray = np.eye(3)
    # linear velocity in world coordinates
    v: np.ndarray = np.zeros((3,))
    # angular velocity in world coordinates
    w: np.ndarray = np.zeros((3,))
    # joint angle
    q: float = 0.0
    # joint velocity
    dq: float = 0.0
    # joint acceleration
    ddq: float = 0.0
    # joint axis vector (relative to parent)
    a: np.ndarray = np.zeros((3,))
    # joint relative position (relative to parent)
    b: np.ndarray = np.zeros((3,))
    # mass
    m: float = 0.0
    # center of mass (link local)
    c: np.ndarray = np.zeros((3,))
    # moment of inertia (link local)
    I: np.ndarray = np.zeros((3, 3))


class ZMPVisualizer:
    def __init__(self):
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.robot: urdf.Robot = urdf.Robot.from_parameter_server()

        self.frame = rospy.get_param("~frame", "odom")
        self.smoothing_window_size = rospy.get_param("~smoothing_window_size", 4)

        self.joint_state_sub = rospy.Subscriber(
            "joint_states",
            JointState,
            callback=self.joint_states_callback,
            queue_size=1,
        )

        self.joint_state_sub = rospy.Subscriber(
            "base_imu",
            Imu,
            callback=self.imu_callback,
            queue_size=1,
        )

        self.exact_zmp_pub = rospy.Publisher("~/zmp/exact", PointStamped, queue_size=1)
        self.approximation_1_zmp_pub = rospy.Publisher(
            "~/zmp/approximation_1", PointStamped, queue_size=1
        )
        self.approximation_2_zmp_pub = rospy.Publisher(
            "~/zmp/approximation_2", PointStamped, queue_size=1
        )
        self.moment_pub = rospy.Publisher("~/moment", PointStamped, queue_size=1)
        self.wrench_pub = rospy.Publisher("~/wrench", WrenchStamped, queue_size=1)

        self.last_P: np.ndarray = None
        self.last_L: np.ndarray = None
        self.last_base_pose: np.ndarray = None
        self.last_base_orientation: np.ndarray = None
        self.last_timestamp: rospy.Time = None

        self.zmp_history: List[np.ndarray] = []

    def joint_states_callback(self, msg: JointState):
        q = dict(zip(msg.name, msg.position))
        dq = dict(zip(msg.name, msg.velocity))
        timestamp = msg.header.stamp
        ulink = self.build_ulink(q, dq)

        if self.last_timestamp is not None and self.last_base_pose is not None:
            dt = timestamp.to_sec() - self.last_timestamp.to_sec()
            ulink.v = (ulink.p - self.last_base_pose) / dt
            # ulink.w = self.compute_angular_velocity(self.last_base_orientation, ulink.R, dt)

        self.forward_velocity(ulink)

        P = self.calc_P(ulink)
        L = self.calc_L(ulink)

        if self.last_timestamp is None or timestamp.to_sec() < self.last_timestamp.to_sec():
            rospy.logwarn("detecting loop, resetting visualizer")
            self.last_timestamp = timestamp
            self.last_P = P
            self.last_L = L
            self.last_base_pose = ulink.p
            self.last_base_orientation = ulink.R
            return

        dt = timestamp.to_sec() - self.last_timestamp.to_sec()
        dP = (P - self.last_P) / dt
        dL = (L - self.last_L) / dt

        wrench = WrenchStamped()
        wrench.header.frame_id = self.frame
        wrench.header.stamp = timestamp
        wrench.wrench.force = msgify(Vector3, dP)
        wrench.wrench.torque = msgify(Vector3, dL)
        self.wrench_pub.publish(wrench)

        m, c = self.compute_com(ulink)
        p_z = self.calc_floor_height(ulink)

        zmp = self.calc_ZMP(m, c, dP, dL, p_z)

        self.zmp_history.append(zmp)

        # smoothing
        zmp_window = np.stack(self.zmp_history[-self.smoothing_window_size:])
        zmp = np.mean(zmp_window, axis=0)

        zmp_msg = PointStamped()
        zmp_msg.header.frame_id = self.frame
        zmp_msg.header.stamp = timestamp
        zmp_msg.point.x = zmp[0]
        zmp_msg.point.y = zmp[1]
        zmp_msg.point.z = zmp[2]
        self.exact_zmp_pub.publish(zmp_msg)

        self.last_timestamp = timestamp
        self.last_P = P
        self.last_L = L
        self.last_base_pose = ulink.p
        self.last_base_orientation = ulink.R

    def compute_com(self, root: Link):
        def crawler(root: Link):
            m = root.m
            c = (root.p + root.R @ root.c) * m

            for child in root.children:
                child_m, child_c = crawler(child)
                m += child_m
                c += child_c
            return m, c

        m, c = crawler(root)
        c /= m
        return m, c

    def forward_velocity(self, root: Link):
        if root.mother is not None:
            mother = root.mother
            root.v = mother.v + np.cross(mother.w, mother.R @ root.b)
            root.w = mother.w + mother.R @ (root.a * root.dq)

        for child in root.children:
            self.forward_velocity(child)

    def calc_P(self, root: Link) -> np.ndarray:
        c1 = root.R @ root.c
        P = root.m * (root.v + np.cross(root.w, c1))
        for child in root.children:
            P += self.calc_P(child)
        return P

    def calc_L(self, root: Link) -> np.ndarray:
        c1 = root.R @ root.c
        c = root.p + c1
        P = root.m * (root.v + np.cross(root.w, c1))
        L = np.cross(c, P) + root.R @ root.I @ root.R.T @ root.w
        for child in root.children:
            L += self.calc_P(child)
        return L

    def calc_ZMP(self, m: float, c: np.ndarray, dP: np.ndarray, dL: np.ndarray, p_z) -> np.ndarray:
        p_x = (m * G * c[0] + p_z * dP[0] - dL[1]) / (m * G + dP[2])
        p_y = (m * G * c[1] + p_z * dP[1] + dL[0]) / (m * G + dP[2])

        return np.array((p_x, p_y, p_z))

    def calc_floor_height(self, root: Link):
        def find_link(root: Link, name: str) -> Link:
            if root.name == name:
                return root

            for child in root.children:
                if (c := find_link(child, name)) is not None:
                    return c

            return None

        left_foot = find_link(root, "left_sole_link").p
        right_foot = find_link(root, "right_sole_link").p

        return min(left_foot[2], right_foot[2])

    def build_ulink(self, q: Dict[str, float], dq: Dict[str, float]) -> Link:
        root_frame: str = self.robot.get_root()

        root_link = Link(root_frame)

        def crawler(root: Link):
            link: urdf.Link = self.robot.link_map[root.name]
            try:
                t: TransformStamped = self.buffer.lookup_transform(
                    self.frame, root.name, rospy.Time(0)
                )
            except tf2_ros.LookupException as e:
                rospy.logerr(f"failed to lookup transform: {e}")
                return

            t: np.ndarray = numpify(t.transform)
            root.R = t[:3, :3]
            root.p = t[:3, 3]

            if link.inertial is not None:
                root.I = np.array(
                    [
                        [
                            link.inertial.inertia.ixx,
                            link.inertial.inertia.ixy,
                            link.inertial.inertia.ixz,
                        ],
                        [
                            link.inertial.inertia.ixy,
                            link.inertial.inertia.iyy,
                            link.inertial.inertia.iyz,
                        ],
                        [
                            link.inertial.inertia.ixz,
                            link.inertial.inertia.iyz,
                            link.inertial.inertia.izz,
                        ],
                    ]
                )
                root.m = link.inertial.mass
                if link.inertial.origin is not None:
                    root.c = np.array(link.inertial.origin.xyz or [0, 0, 0])

            if root.name in self.robot.child_map:
                for joint_name, child_name in self.robot.child_map[root.name]:
                    child_link: urdf.Link = self.robot.link_map[child_name]
                    child = Link(child_link.name)
                    joint: urdf.Joint = self.robot.joint_map[joint_name]
                    child.a = np.array(joint.axis or [0, 0, 0])
                    if joint.origin is not None:
                        # TODO: maybe the offset is in the wrong coordinate frame?
                        child.b = np.array(joint.origin.xyz or [0, 0, 0])
                    if joint_name in dq:
                        child.q = q[joint_name]
                        child.dq = dq[joint_name]
                    child.mother = root
                    crawler(child)
                    root.children.append(child)

        crawler(root_link)

        return root_link

    def compute_angular_velocity(self, R1, R2, dt):
        # Compute the relative rotation matrix
        R_rel = np.dot(R2, R1.T)

        # Extract the angular velocity vector
        omega_x = (R_rel[2, 1] - R_rel[1, 2]) / (2 * dt)
        omega_y = (R_rel[0, 2] - R_rel[2, 0]) / (2 * dt)
        omega_z = (R_rel[1, 0] - R_rel[0, 1]) / (2 * dt)

        omega = np.array([omega_x, omega_y, omega_z])

        return omega

    def imu_callback(self, msg: Imu):
        ulink = self.build_ulink({}, {})
        m, com = self.compute_com(ulink)

        try:
            t: TransformStamped = self.buffer.lookup_transform(
                self.frame, msg.header.frame_id, rospy.Time(0)
            )
        except tf2_ros.LookupException as e:
            rospy.logerr(f"failed to lookup transform: {e}")
            return

        t: np.ndarray = numpify(t.transform)
        accel = numpify(msg.linear_acceleration)
        accel = t[:3, :3] @ accel

        p_z = self.calc_floor_height(ulink)
        p_x = com[0] - (com[2] - p_z) * accel[0] / accel[2]  # gravity already in imu measurement
        p_y = com[1] - (com[2] - p_z) * accel[1] / accel[2]  # gravity already in imu measurement

        zmp_msg = PointStamped()
        zmp_msg.header.frame_id = self.frame
        zmp_msg.header.stamp = msg.header.stamp
        zmp_msg.point.x = p_x
        zmp_msg.point.y = p_y
        zmp_msg.point.z = p_z
        self.approximation_2_zmp_pub.publish(zmp_msg)


def main():
    rospy.init_node("zmp_visualizer")

    node = ZMPVisualizer()
    rospy.loginfo("zmp_visualizer initialized")
    rospy.spin()


if __name__ == "__main__":
    main()
