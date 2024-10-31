#!/usr/bin/env python
import logging

import time
import scipy
from horizon.ros import utils as horizon_ros_utils
from ttictoc import tic,toc
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32
import numpy as np
import keyboard
import rospy
import casadi as cs
from srbd_horizon import ModelSchedulingMpc


horizon_ros_utils.roslaunch("srbd_horizon", "SRBD_kangaroo_line_feet.launch")
time.sleep(3.)

# creates HORIZON problem, these parameters can not be tuned at the moment
ns = {'SRBD': 10, 'LIP': 10}
T = {'SRBD': 0.5, 'LIP': 0.5}

joint_init = rospy.get_param("joint_init")
if len(joint_init) == 0:
    print("joint_init parameter is mandatory, exiting...")
    exit()

dmodel_scheduling_mpc = ModelSchedulingMpc.ModelSchedulingController(joint_init, ns, T)

rate = rospy.Rate(rospy.get_param("hz", 10)) # 10 Hz
while not rospy.is_shutdown():
    solution = dmodel_scheduling_mpc.get_solution(state=None)
    rate.sleep()