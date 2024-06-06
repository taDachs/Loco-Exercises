#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, Transform
import rospy
from typing import List, Dict, cast
from ros_numpy import numpify
import urdf_parser_py.urdf as urdf

from dataclasses import dataclass

ARM_LINK_NAMES = [
    "torso_1_link",
    "torso_2_link",
    "arm_right_1_link",
    "arm_right_2_link",
    "arm_right_3_link",
    "arm_right_4_link",
    "arm_right_5_link",
    "arm_right_6_link",
    "arm_right_7_link",
]


@dataclass
class Link:
    R: np.ndarray
    p: np.ndarray
    a: np.ndarray = None
    mother: str = None


def build_ulink() -> List[Link]:
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)

    robot: urdf.Robot = urdf.Robot.from_parameter_server()

    rospy.loginfo("Sleeping for 2 seconds to make sure tf buffer has something")
    rospy.sleep(2)

    ulink: Dict[str, Link] = {}

    global_frame = robot.get_root()

    for link in robot.links:
        link: urdf.Link = cast(urdf.Link, link)

        try:
            t: TransformStamped = buffer.lookup_transform(global_frame, link.name, rospy.Time(0))
        except tf2_ros.LookupException as e:
            rospy.logerr(f"failed to lookup transform: {e}")
            return

        t: np.ndarray = numpify(t.transform)
        R = t[:3, :3]
        p = t[:3, 3]
        ulink[link.name] = Link(R=R, p=p)

    for joint_name, joint in robot.joint_map.items():
        joint: urdf.Joint = cast(urdf.Joint, joint)

        ulink[joint.child].mother = joint.parent
        ulink[joint.child].a = np.array(joint.axis)

    return ulink


def calc_jacobian(ulink: Dict[str, Link], idx: List[str]):
    target = ulink[idx[-1]].p
    J = np.zeros((6, len(idx)))

    for n, link_name in enumerate(idx):
        j = ulink[link_name]
        if j.mother is not None:
            mom = ulink[j.mother]
            a = mom.R @ j.a
        else:
            a = j.a

        J[:, n] = np.concatenate([np.cross(a, target - j.p), a])
    return J


def compute_manipulability(J):
    J_J_T = J @ J.T
    eigen_values, _ = np.linalg.eig(J_J_T)

    mu_1 = np.sqrt(np.max(eigen_values) / np.min(eigen_values))

    mu_2 = np.max(eigen_values) / np.min(eigen_values)

    mu_3 = np.sqrt(np.prod(eigen_values))

    return mu_1, mu_2, mu_3


def main():
    rospy.init_node("jacobian_calculator", anonymous=True)
    ulink = build_ulink()
    J = calc_jacobian(ulink, ARM_LINK_NAMES)
    print(f"J: {J}")
    mu_1, mu_2, mu_3 = compute_manipulability(J)
    print("Manipulability:")
    print(
        "\tRatio of the longest and shortest semi-axes of the manipulability ellipsoid (smaller "
        f"is better):  {mu_1}"
    )
    print(f"\tCondition Number (smaller_is_better): {mu_2}")
    print(f"\tVolume of the manipulability ellipsoid (larger is better): {mu_3}")


if __name__ == "__main__":
    main()
