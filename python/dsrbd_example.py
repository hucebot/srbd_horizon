#!/usr/bin/env python
import logging

import time
from horizon.ros import utils as horizon_ros_utils
from ttictoc import tic,toc
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32
import viz
import wpg
import cartesio #todo: use bindings!
import numpy as np
import keyboard
import rospy
import prb as srbd_problem
import casadi as cs
from horizon.transcriptions import integrators
import utilities
from horizon.utils import utils, kin_dyn
def joy_cb(msg):
    global joy_msg
    joy_msg = msg


horizon_ros_utils.roslaunch("srbd_horizon", "SRBD_kangaroo_line_feet.launch")
time.sleep(3.)

# creates HORIZON problem, these parameters can not be tuned at the moment
ns = 20
T = 1.

srbd = srbd_problem.SRBDProblem()
srbd.createSRBDProblem(ns, T)

# create solver
max_iteration = rospy.get_param("max_iteration", 20)
print(f"max_iteration: {max_iteration}")

initial_state = np.array([float(srbd.com[0]), float(srbd.com[1]), float(srbd.com[2]),
                          0.,     0.,   0.,   1.,
                          float(srbd.initial_foot_position[0][0]),  float(srbd.initial_foot_position[0][1]),   float(srbd.initial_foot_position[0][2]),
                          float(srbd.initial_foot_position[1][0]),  float(srbd.initial_foot_position[1][1]),   float(srbd.initial_foot_position[1][2]),
                          float(srbd.initial_foot_position[2][0]),  float(srbd.initial_foot_position[2][1]),   float(srbd.initial_foot_position[2][2]),
                          float(srbd.initial_foot_position[3][0]),  float(srbd.initial_foot_position[3][1]),   float(srbd.initial_foot_position[3][2]),
                          0.,     0.,   0.,
                          0.,     0.,   0.,
                          0.,     0.,   0.,
                          0.,     0.,   0.,
                          0.,     0.,   0.,
                          0.,     0.,   0.])
static_input = np.array([0., 0., 0., 0., 0., srbd.m * 9.81 / srbd.force_scaling / 4,
                         0., 0., 0., 0., 0., srbd.m * 9.81 / srbd.force_scaling / 4,
                         0., 0., 0., 0., 0., srbd.m * 9.81 / srbd.force_scaling / 4,
                         0., 0., 0., 0., 0., srbd.m * 9.81 / srbd.force_scaling / 4])

rospy.init_node('srbd_mpc_test', anonymous=True)

solution_time_pub = rospy.Publisher("solution_time", Float32, queue_size=10)
srbd_pub = rospy.Publisher("srbd_constraint", WrenchStamped, queue_size=10)
srbd_msg = WrenchStamped()

# game controller
rate = rospy.Rate(rospy.get_param("hz", 10)) # 10 Hz
rospy.Subscriber('/joy', Joy, joy_cb)
global joy_msg
joy_msg = None


import ddp
opts = dict()
opts["max_iters"] = 100
opts["alpha_converge_threshold"] = 1e-12
opts["beta"] = 1e-3
solver = ddp.DDPSolver(srbd.prb, opts=opts)

# set initial state and warmstart ddp
state = initial_state
x_warmstart = np.zeros((state.shape[0], ns+1))
for i in range(0, ns+1):
    x_warmstart[:, i] = state
u_warmstart = np.zeros((static_input.shape[0], ns))
for i in range(0, ns):
    u_warmstart[:, i] = static_input

#define discrete dynamics
dae = dict()
dae["x"] = cs.vertcat(srbd.prb.getState().getVars())
dae["ode"] = srbd.prb.getDynamics()
dae["p"] = cs.vertcat(srbd.prb.getInput().getVars())
dae["quad"] = 0.
simulation_euler_integrator = integrators.EULER(dae)

# Walking patter generator and scheduler
wpg = wpg.steps_phase(srbd.f, srbd.c, srbd.cdot, srbd.initial_foot_position[0][2].__float__(), srbd.c_ref, srbd.w_ref, srbd.orientation_tracking_gain, srbd.cdot_switch, ns, number_of_legs=2,
                      contact_model=srbd.contact_model)
ci = cartesio.cartesIO(["left_sole_link", "right_sole_link"])
while not rospy.is_shutdown():

    solver.setInitialState(state)

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
        srbd.rdot_ref.assign(srbd.rdot_ref.getValues(nodes=j), nodes=j-1)
        srbd.w_ref.assign(srbd.w_ref.getValues(nodes=j), nodes=j-1)
        srbd.oref.assign(srbd.oref.getValues(nodes=j), nodes=j - 1)
        srbd.orientation_tracking_gain.assign(srbd.orientation_tracking_gain.getValues(nodes=j), nodes=j - 1)

    # assign new references based on user input
    if motion == "standing":
        alphaX, alphaY = 0.1, 0.1
    else:
        alphaX, alphaY = 0.5, 0.5

    if joy_msg is not None:
        srbd.rdot_ref.assign([alphaX * joy_msg.axes[1], alphaY * joy_msg.axes[0], 0.1 * joy_msg.axes[7]], nodes=ns)
        #w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]], nodes=ns)
        #orientation_tracking_gain.assign(cs.sqrt(1e5) if rotate else 0.)
    else:
        axis_x = keyboard.is_pressed('up') - keyboard.is_pressed('down')
        axis_y = keyboard.is_pressed('right') - keyboard.is_pressed('left')

        srbd.rdot_ref.assign([alphaX * axis_x, alphaY * axis_y, 0], nodes=ns)
        #w_ref.assign([0, 0, 0], nodes=ns)
        #orientation_tracking_gain.assign(0.)

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

    c0_hist = dict()
    for i in range(0, srbd.nc):
        c0_hist['c' + str(i)] = solution['c' + str(i)][:, 0]

    t = rospy.Time().now()
    utilities.SRBDTfBroadcaster(solution['r'][:, 0], solution['o'][:, 0], c0_hist, t)
    for i in range(0, srbd.nc):
        viz.publishContactForce(t, srbd.force_scaling * solution['f' + str(i)][:, 0], 'c' + str(i))
        viz.publishPointTrj(solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
    viz.SRBDViewer(srbd.I, "SRB", t, srbd.nc)  # TODO: should we use w_R_b * I * w_R_b.T?
    viz.publishPointTrj(solution["r"], t, "SRB", "world")

    cc = dict()
    ff = dict()
    for i in range(0, srbd.nc):
        cc[i] = solution["c" + str(i)][:, 0]
        ff[i] = solution["f" + str(i)][:, 0]

    # simulation integration
    input = solution["u_opt"][:, 0]
    state = simulation_euler_integrator(state, input, srbd.prb.getDt())[0]
    state[3:7] /= cs.norm_2(state[3:7])
    #print(f"state:", solution["x_opt"])
    #print(f"input:", solution["u_opt"])
    rddot0 = srbd.RDDOT(input)
    wdot0 = srbd.WDOT(state, input)

    w_R_b0 = utils.toRot(state[3:7])
    srbd_0 = kin_dyn.SRBD(srbd.m/srbd.force_scaling, w_R_b0*srbd.I/srbd.force_scaling*w_R_b0.T, ff, solution["r"][:, 0], rddot0, cc, solution["w"][:, 0], wdot0)
    srbd_msg.header.stamp = t
    srbd_msg.wrench.force.x = srbd_0[0]
    srbd_msg.wrench.force.y = srbd_0[1]
    srbd_msg.wrench.force.z = srbd_0[2]
    srbd_msg.wrench.torque.x = srbd_0[3]
    srbd_msg.wrench.torque.y = srbd_0[4]
    srbd_msg.wrench.torque.z = srbd_0[5]
    srbd_pub.publish(srbd_msg)

    ci.publish(solution["r"][:, 1], solution["rdot"][:, 1],
               solution["o"][:, 1], solution["w"][:, 1],
               {"left_sole_link": [solution['c' + str(0)][:, 1], solution['c' + str(1)][:, 1]],
                "right_sole_link": [solution['c' + str(2)][:, 1], solution['c' + str(3)][:, 1]]},
               {"left_sole_link": [solution['cdot' + str(0)][:, 1], solution['cdot' + str(1)][:, 1]],
                "right_sole_link": [solution['cdot' + str(2)][:, 1], solution['cdot' + str(3)][:, 1]]},
               t)

    rate.sleep()
