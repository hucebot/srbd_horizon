#!/usr/bin/env python

import rospy
from horizon.ros import utils as horizon_ros_utils

import time
from srbd_horizon import mpc


horizon_ros_utils.roslaunch("srbd_horizon", "full_model_kangaroo.launch")
time.sleep(3.)

# creates HORIZON problem, these parameters can not be tuned at the moment
ns = 20
T = 1.

joint_init = rospy.get_param("joint_init")
if len(joint_init) == 0:
    print("joint_init parameter is mandatory, exiting...")
    exit()

full_model_mpc = mpc.fullModelController(joint_init, ns, T)

# game controller
rate = rospy.Rate(rospy.get_param("hz", 10)) # 10 Hz
while not rospy.is_shutdown():
    solution = full_model_mpc.get_solution(state=None)
    rate.sleep()


