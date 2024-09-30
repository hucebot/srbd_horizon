#!/usr/bin/env python

import ddp
import rospy
from horizon.ros import utils as horizon_ros_utils
from horizon.utils import mat_storer
import time
import prb
from sensor_msgs.msg import Joy, JointState
import numpy as np
from ttictoc import tic,toc
from std_msgs.msg import Float32
import tf
import viz
import wpg
import keyboard
import utilities

def joy_cb(msg):
    global joy_msg
    joy_msg = msg

horizon_ros_utils.roslaunch("srbd_horizon", "full_model_kangaroo.launch")
time.sleep(3.)

# creates HORIZON problem, these parameters can not be tuned at the moment
ns = 20
T = 1.

full_model = prb.FullBodyProblem()
full_model.createFullBodyProblem(ns, T, include_transmission_forces=False)

# create solver
max_iteration = rospy.get_param("max_iteration", 1)


rospy.init_node('full_mpc_test', anonymous=True)
solution_time_pub = rospy.Publisher("solution_time", Float32, queue_size=10)
joint_state_publisher = rospy.Publisher("joint_states", JointState, queue_size=10)

# game controller
rate = rospy.Rate(rospy.get_param("hz", 10)) # 10 Hz
rospy.Subscriber('/joy', Joy, joy_cb)
global joy_msg
joy_msg = None

import ddp

opts = {"gnsqp.max_iter": max_iteration,
         'gnsqp.osqp.scaled_termination': False,
         'gnsqp.eps_regularization': 1e-3, #1e-2,
        'gnsqp.osqp.polish': False,
         'gnsqp.osqp.verbose': False}

solver = ddp.SQPSolver(full_model.prb, qp_solver_plugin='osqp', opts=opts)
full_model.q.setInitialGuess(full_model.getInitialState()[0:full_model.nq])
full_model.qdot.setInitialGuess(full_model.getInitialState()[full_model.nq:])
full_model.qddot.setInitialGuess(full_model.getStaticInput()[0:full_model.nv])
i = -1
for foot_frame in full_model.foot_frames:
    i += 1
    full_model.f[foot_frame].setInitialGuess(full_model.getStaticInput()[full_model.nv+i*3:full_model.nv+i*3+3])
if full_model.include_transmission_forces:
    full_model.left_actuation_lambda.setInitialGuess(full_model.getStaticInput()[full_model.nv+i*3+3:full_model.nv+i*3+3+2])
    full_model.right_actuation_lambda.setInitialGuess(full_model.getStaticInput()[full_model.nv+i*3+3+2:])

solver.setInitialGuess(full_model.getInitialGuess())

"""
Dictionary to store variables used for warm-start
"""
variables_dict = {"q": full_model.q, "qdot": full_model.qdot, "qddot": full_model.qddot}
if full_model.include_transmission_forces:
    variables_dict["left_actuation_lambda"] = full_model.left_actuation_lambda
    variables_dict["right_actuation_lambda"] = full_model.right_actuation_lambda
for foot_frame in full_model.foot_frames:
    variables_dict["f_" + foot_frame] = full_model.f[foot_frame]


solver.solve()
solution = solver.getSolutionDict()
solution['q'] = utilities.normalize_quaternion_part_horizon(solution['q'], ns)

joint_state_msg = JointState()
joint_state_msg.name = full_model.kindyn.joint_names()[2:]


# create data structures for wpg: copying data to different keys indexed by id
k = 0
f = dict()
c = dict()
cdot = dict()
initial_foot_position = dict()
c_ref = dict()
#cdot_switch = dict()
cdotxy_tracking_constraint = dict()
for foot_frame in full_model.foot_frames:
    f[k] = full_model.f[foot_frame]
    c[k] = full_model.c[foot_frame]
    cdot[k] = full_model.cdot[foot_frame]
    initial_foot_position[k] = full_model.initial_foot_position[foot_frame]
    c_ref[k] = full_model.c_ref[foot_frame]
    #cdot_switch[k] = full_model.cdot_switch[foot_frame]
    cdotxy_tracking_constraint[k] = full_model.cdotxy_tracking_constraint[foot_frame]
    k += 1


wpg = wpg.steps_phase(f, c, cdot, initial_foot_position[0][2].__float__(), c_ref, full_model.w_ref,
                      full_model.orientation_tracking_gain, cdot_switch=None, nodes=ns, number_of_legs=2,
                      contact_model=full_model.contact_model, cdotxy_tracking_constraint=cdotxy_tracking_constraint)

while not rospy.is_shutdown():
    """
    Automatically set initial guess from solution to variables in variables_dict
    """
    mat_storer.setInitialGuess(variables_dict, solution)
    solver.setInitialGuess(full_model.getInitialGuess())
    #open loop
    full_model.q.setBounds(solution['q'][:, 1], solution['q'][:, 1], 0)
    full_model.qdot.setBounds(solution['qdot'][:, 1], solution['qdot'][:, 1], 0)


    # references
    motion = "standing"
    rotate = False
    if joy_msg is not None:
        if joy_msg.buttons[4]:
            motion = "walking"
        elif joy_msg.buttons[5]:
            motion = "jumping"
        if joy_msg.buttons[3]:
            rotate = True
    else:
        if keyboard.is_pressed('ctrl'):
            motion = "walking"
        if keyboard.is_pressed('space'):
            motion = "jumping"

    # shift reference velocities back by one node
    for j in range(1, ns + 1):
        full_model.rdot_ref.assign(full_model.rdot_ref.getValues(nodes=j), nodes=j - 1)
        full_model.w_ref.assign(full_model.w_ref.getValues(nodes=j), nodes=j - 1)
        full_model.oref.assign(full_model.oref.getValues(nodes=j), nodes=j - 1)
        full_model.orientation_tracking_gain.assign(full_model.orientation_tracking_gain.getValues(nodes=j), nodes=j - 1)

    # assign new references based on user input
    if motion == "standing":
        alphaX, alphaY = 0.1, 0.1
    else:
        alphaX, alphaY = 0.5, 0.5

    if joy_msg is not None:
        full_model.rdot_ref.assign([alphaX * joy_msg.axes[1], alphaY * joy_msg.axes[0], 0.1 * joy_msg.axes[7]], nodes=ns)
        # w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]], nodes=ns)
        # orientation_tracking_gain.assign(cs.sqrt(1e5) if rotate else 0.)
    else:
        axis_x = keyboard.is_pressed('up') - keyboard.is_pressed('down')
        axis_y = keyboard.is_pressed('right') - keyboard.is_pressed('left')

        full_model.rdot_ref.assign([alphaX * axis_x, alphaY * axis_y, 0], nodes=ns)
        # w_ref.assign([0, 0, 0], nodes=ns)
        # orientation_tracking_gain.assign(0.)

    if motion == "walking":
        wpg.set("step")
    elif motion == "jumping":
        wpg.set("jump")
    else:
        wpg.set("standing")

    # solve
    tic()
    solver.solve()
    solution_time_pub.publish(toc())
    solution = solver.getSolutionDict()
    solution['q'] = utilities.normalize_quaternion_part_horizon(solution['q'], ns)

    t = rospy.Time.now()
    # publish tf
    br = tf.TransformBroadcaster()
    br.sendTransform(solution['q'][0:3, 1], solution['q'][3:7, 1], t, "base_link", "world")

    # publish joint states
    joint_state_msg.position = solution['q'][7:, 1]
    joint_state_msg.header.stamp = t
    joint_state_publisher.publish(joint_state_msg)

    # publish contact forces and contact points
    c = dict()
    for foot_frame in full_model.foot_frames:
        C = full_model.kindyn.fk(foot_frame)
        c[foot_frame] = np.zeros((3, ns + 1))
        for i in range(0, ns + 1):
            c[foot_frame][:, i] = C(q=solution['q'][:, i])['ee_pos'].toarray().flatten()
    for i in range(0, full_model.nc):
        viz.publishContactForce(t, solution['f_' + full_model.foot_frames[i]][:, 0], frame=full_model.foot_frames[i], topic='fc' + str(i))
        viz.publishPointTrj(c[full_model.foot_frames[i]], t, 'c' + str(i), "world", color=[0., 0., 1.])

    # publish center of mass
    COM = full_model.kindyn.centerOfMass()
    com = np.zeros((3, ns+1))
    for i in range(0, ns+1):
        com[:, i] = COM(q=solution['q'][:, i])['com'].toarray().flatten()
    viz.publishPointTrj(com, t, "SRB", "world")

    rate.sleep()


