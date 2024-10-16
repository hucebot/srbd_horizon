#!/usr/bin/env python
import logging

import time
import scipy
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
lip = srbd_problem.LIPProblem()
lip.createLIPProblem(ns, T)

# create solver
max_iteration = rospy.get_param("max_iteration", 20)
print(f"max_iteration: {max_iteration}")


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
solver = ddp.DDPSolver(lip.prb, opts=opts)

# set initial state and warmstart ddp
state = lip.getInitialState()
x_warmstart = np.zeros((state.shape[0], ns+1))
for i in range(0, ns+1):
    x_warmstart[:, i] = state
u_warmstart = np.zeros((lip.getStaticInput().shape[0], ns))
for i in range(0, ns):
    u_warmstart[:, i] = lip.getStaticInput()

np.set_printoptions(suppress=True)

#define discrete dynamics
dae = dict()
dae["x"] = cs.vertcat(lip.prb.getState().getVars())
dae["ode"] = lip.prb.getDynamics()
dae["p"] = cs.vertcat(lip.prb.getInput().getVars())
dae["quad"] = 0.
simulation_euler_integrator = solver.get_f(0)

# Walking patter generator and scheduler
wpg = wpg.steps_phase(number_of_legs=2, contact_model=lip.contact_model, c_init_z=lip.initial_foot_position[0][2].__float__())

ci = cartesio.cartesIO(["left_sole_link", "right_sole_link"])
solution_time_vec = list()
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
        lip.rdot_ref.assign(lip.rdot_ref.getValues(nodes=j), nodes=j-1)
        lip.eta2_p.assign(lip.eta2_p.getValues(nodes=j), nodes=j-1)

    if lip.cdot_switch[0].getValues(ns) == 0 and lip.cdot_switch[1].getValues(ns) == 0 and lip.cdot_switch[2].getValues(ns) == 0 and lip.cdot_switch[3].getValues(ns) == 0:
        lip.eta2_p.assign(0., nodes=ns)
    else:
        lip.eta2_p.assign(lip.eta2, nodes=ns)


    # assign new references based on user input
    if motion == "standing":
        alphaX, alphaY = 0.1, 0.1
    else:
        alphaX, alphaY = 0.5, 0.5

    if joy_msg is not None:
        lip.rdot_ref.assign([alphaX * joy_msg.axes[1], alphaY * joy_msg.axes[0], 0.1 * joy_msg.axes[7]], nodes=ns)
        #w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]], nodes=ns)
        #orientation_tracking_gain.assign(cs.sqrt(1e5) if rotate else 0.)
    else:
        axis_x = keyboard.is_pressed('up') - keyboard.is_pressed('down')
        axis_y = keyboard.is_pressed('right') - keyboard.is_pressed('left')

        lip.rdot_ref.assign([alphaX * axis_x, alphaY * axis_y, 0], nodes=ns)
        #w_ref.assign([0, 0, 0], nodes=ns)
        #orientation_tracking_gain.assign(0.)

    lip.shiftContactConstraints()
    lip.setAction(motion, wpg)

    # solve
    tic()
    solver.solve()
    solution_time = toc()
    solution_time_vec.append(solution_time)
    solution_time_pub.publish(solution_time)
    solution = solver.getSolutionDict()

    c0_hist = dict()
    for i in range(0, lip.nc):
        c0_hist['c' + str(i)] = solution['c' + str(i)][:, 0]

    t = rospy.Time().now()
    utilities.SRBDTfBroadcaster(solution['r'][:, 0], np.array([0., 0., 0., 1.]), c0_hist, t)
    utilities.ZMPTfBroadcaster(solution['z'][:, 0], t)

    input = solution["u_opt"][:, 0]
    state = np.array(cs.DM(simulation_euler_integrator(state, input, solver.get_params_value(0))))

    #print(f"state {state}")
    #print(f"input {input}")
    #print(f"solver.get_params_value(0) {solver.get_params_value(0)}")
    rddot0 = lip.RDDOT(state, input, solver.get_params_value(0))
    fzmp = lip.m * (np.array([0., 0., 9.81]) + rddot0)
    viz.publishContactForce(t, fzmp, 'ZMP')
    for i in range(0, lip.nc):
        viz.publishPointTrj(solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
    viz.SRBDViewer(srbd.I, "SRB", t, lip.nc)  # TODO: should we use w_R_b * I * w_R_b.T?
    viz.publishPointTrj(solution["r"], t, "SRB", "world")
    viz.publishPointTrj(solution["z"], t, name="ZMP", frame="world", color=[0., 1., 1.], namespace="LIP")

    cc = dict()
    for i in range(0, lip.nc):
        cc[i] = solution["c" + str(i)][:, 0]

    # simulation integration
    #print(f"state:", solution["x_opt"])
    #print(f"input:", solution["u_opt"])
    #rddot0 = lip.RDDOT(input)


    w_R_b0 = utils.toRot(state[3:7])
    #srbd_0 = kin_dyn.SRBD(lip.m/lip.force_scaling, w_R_b0*lip.I/lip.force_scaling*w_R_b0.T, ff, solution["r"][:, 0], rddot0, cc, solution["w"][:, 0], wdot0)
    srbd_msg.header.stamp = t
    #srbd_msg.wrench.force.x = srbd_0[0]
    #srbd_msg.wrench.force.y = srbd_0[1]
    #srbd_msg.wrench.force.z = srbd_0[2]
    #srbd_msg.wrench.torque.x = srbd_0[3]
    #srbd_msg.wrench.torque.y = srbd_0[4]
    #srbd_msg.wrench.torque.z = srbd_0[5]
    srbd_pub.publish(srbd_msg)

    ci.publish(solution["r"][:, 1], solution["rdot"][:, 1],
               np.array([0., 0., 0., 1.]), np.array([0., 0., 0.]),
               {"left_sole_link": [solution['c' + str(0)][:, 1], solution['c' + str(1)][:, 1]],
                "right_sole_link": [solution['c' + str(2)][:, 1], solution['c' + str(3)][:, 1]]},
               {"left_sole_link": [solution['cdot' + str(0)][:, 1], solution['cdot' + str(1)][:, 1]],
                "right_sole_link": [solution['cdot' + str(2)][:, 1], solution['cdot' + str(3)][:, 1]]},
               t)

    rate.sleep()

scipy.io.savemat('dlip_solution_time.mat', {'solution_time': np.array(solution_time_vec)})
