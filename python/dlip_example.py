#!/usr/bin/env python
import logging

import time
from horizon.ros import utils as horizon_ros_utils
import rospy
import mpc
def joy_cb(msg):
    global joy_msg
    joy_msg = msg


horizon_ros_utils.roslaunch("srbd_horizon", "SRBD_kangaroo_line_feet.launch")
time.sleep(3.)

# creates HORIZON problem, these parameters can not be tuned at the moment
ns = 20
T = 1.

joint_init = rospy.get_param("joint_init")
if len(joint_init) == 0:
    print("joint_init parameter is mandatory, exiting...")
    exit()

lip_mpc = mpc.LipController(joint_init, ns, T)

rate = rospy.Rate(rospy.get_param("hz", 10))  # 10 Hz
while not rospy.is_shutdown():
    state, input, rddot0, fzmp = lip_mpc.get_solution(state=None)

    rate.sleep()
