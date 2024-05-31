#!/usr/bin/env python3
from __future__ import annotations

import rospy
from geometry_msgs.msg import PoseStamped
import sys
import moveit_commander
import tf2_ros
import tf2_geometry_msgs
import argparse


def execute_trajectory(pose: PoseStamped, group_name: str, use_orientation: bool = True):
    group = moveit_commander.MoveGroupCommander(group_name)

    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)

    rospy.loginfo("Sleeping for 2 seconds to make sure tf buffer has something")
    rospy.sleep(2)

    # We can get the name of the reference frame for this robot:
    planning_frame = group.get_planning_frame()

    tr = buffer.lookup_transform(planning_frame, pose.header.frame_id, rospy.Time(0))
    pose = tf2_geometry_msgs.do_transform_pose(pose, tr)

    if use_orientation:
        rospy.loginfo(f"Moving {group.get_end_effector_link()} to {pose.pose}")
        group.set_pose_target(pose.pose)
    else:
        position = [
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ]
        rospy.loginfo(f"Moving {group.get_end_effector_link()} to {position}")
        group.set_position_target(position)
    result = group.go(wait=True)

    if result:
        rospy.loginfo("Movement successful")
    else:
        rospy.logerr("Movement failed")

    group.stop()
    group.clear_pose_targets()


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_ik", anonymous=True)

    parser = argparse.ArgumentParser("moveit_ik")
    parser.add_argument("--position", nargs=3, type=float, required=True)
    parser.add_argument("--orientation", nargs=4, type=float, required=False)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--frame", default="base_link")
    args = parser.parse_args()

    pose_goal = PoseStamped()
    pose_goal.header.frame_id = args.frame
    pose_goal.pose.position.x = args.position[0]
    pose_goal.pose.position.y = args.position[1]
    pose_goal.pose.position.z = args.position[2]

    if args.orientation is not None:
        pose_goal.pose.orientation.x = args.orientation[0]
        pose_goal.pose.orientation.y = args.orientation[1]
        pose_goal.pose.orientation.z = args.orientation[2]
        pose_goal.pose.orientation.w = args.orientation[3]

    execute_trajectory(pose_goal, args.group, use_orientation=args.orientation is not None)


if __name__ == "__main__":
    main()
