#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#from __future__ import print_function
import rospy, time
import subprocess #import Popen

from numpy import pi

from std_msgs.msg import String

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    FollowJointTrajectoryResult,
)

from trajectory_msgs.msg import (     
    JointTrajectory,
    JointTrajectoryPoint,
)

rospy.init_node('facepalmer', anonymous=True)
rate = rospy.Rate(1)

controller='/right_arm_controller'

import actionlib

client=actionlib.SimpleActionClient(controller + "/follow_joint_trajectory", FollowJointTrajectoryAction)
client.wait_for_server()


right_arm_joint_names = rospy.get_param(controller+"/joints")
print(right_arm_joint_names)

time.sleep(3)
print("let's start")

target = JointTrajectoryPoint()
#target.positions = [0.1]*len(right_arm_joint_names)
target.positions=[
    pi/2, # rotation forward shoulder
    0.2,    # rotation sidewards shoulder
    -1.0,    # rotation arm-internal
    pi/2+0.4,  # rotation ellbow
    2,
    -0.1,
    0
]
target.time_from_start = rospy.Duration(5)  # seconds

goal = FollowJointTrajectoryGoal()
goal.trajectory.points.append(target)
goal.goal_time_tolerance=rospy.Duration(0.5)
goal.trajectory.joint_names=right_arm_joint_names

i=0
while not rospy.is_shutdown() and i < 3:
    client.send_goal(goal)
    client.wait_for_result(timeout=rospy.Duration(60))
    #rate.sleep()
    print("Goal sent")
    time.sleep(10)
    i+=1
