#!/usr/bin/env python
import logging

import rospy
import casadi as cs
import numpy as np
from horizon import problem, variables
from horizon.utils import utils, kin_dyn, resampler_trajectory, mat_storer
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.transcriptions import integrators
from horizon.solvers import solver
from horizon.ros.replay_trajectory import *
import time
from horizon.ros import utils as horizon_ros_utils
from ttictoc import tic,toc
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32
import viz
import wpg
import utilities
import cartesio #todo: use bindings!
import solver_options

import lip
import keyboard

def joy_cb(msg):
    global joy_msg
    joy_msg = msg


horizon_ros_utils.roslaunch("srbd_horizon", "SRBD_kangaroo_line_feet.launch")
time.sleep(3.)

# creates HORIZON problem, these parameters can not be tuned at the moment
ns = 20
prb = problem.Problem(ns, casadi_type=cs.SX)
T = 1.


urdf = rospy.get_param("robot_description", "")
if urdf == "":
    print("robot_description not loaded in param server!")
    exit()

kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# create variables

r = prb.createStateVariable("r", 3) # com position
o = prb.createStateVariable("o", 4) # base orientation quaternion
q = variables.Aggregate()           # position aggregate
q.addVariable(r)
q.addVariable(o)

# contacts position
contact_model = rospy.get_param("contact_model", 4)
number_of_legs = rospy.get_param("number_of_legs", 2)
nc = number_of_legs * contact_model

c = dict()
for i in range(0, nc):
    c[i] = prb.createStateVariable("c" + str(i), 3) # Contact i position
    q.addVariable(c[i])

# variables
rdot = prb.createStateVariable("rdot", 3) # com velocity
w = prb.createStateVariable("w", 3)       # base vel
qdot = variables.Aggregate()              # velocity aggregate
qdot.addVariable(rdot)
qdot.addVariable(w)

# contacts velocity
cdot = dict()
for i in range(0, nc):
    cdot[i] = prb.createStateVariable("cdot" + str(i), 3)  # Contact i vel
    qdot.addVariable(cdot[i])

# variable to collect all acceleration controls
qddot = variables.Aggregate()

cddot = dict()
f = dict()
for i in range(0, nc):
    cddot[i] = prb.createInputVariable("cddot" + str(i), 3) # Contact i acc
    f[i] = prb.createInputVariable("f" + str(i), 3) # Contact i forces

# references
rdot_ref = prb.createParameter('rdot_ref', 3)
w_ref = prb.createParameter('w_ref', 3)

rdot_ref.assign([0., 0., 0.], nodes=range(1, ns+1))
w_ref.assign([0., 0., 0.], nodes=range(1, ns+1))

# Formulate discrete time dynamics using multiple_shooting and RK2 integrator
# joint_init is used to initialize the urdf model and retrieve information such as: CoM, Inertia, atc... 
# at the nominal configuration given by joint_init

joint_init = rospy.get_param("joint_init")
if len(joint_init) == 0:
    print("joint_init parameter is mandatory, exiting...")
    exit()

if rospy.has_param("world_frame_link"):
    world_frame_link = rospy.get_param("world_frame_link")
    utilities.setWorld(world_frame_link, kindyn, joint_init)
    print(f"world_frame_link: {world_frame_link}")

print(f"joint_init: {joint_init}")
m = kindyn.mass()
print(f"mass: {m}")
M = kindyn.crba()
I = M(q=joint_init)['B'][3:6, 3:6]
print(f"I centroidal in base: {I}")
w_R_b = utils.toRot(o)
force_scaling = 1000.
rddot, wdot = kin_dyn.fSRBD(m/force_scaling, w_R_b * (I/force_scaling) * w_R_b.T, f, r, c, w) #scaled forces

RDDOT = cs.Function('rddot', [prb.getInput().getVars()], [rddot])
WDOT = cs.Function('wdot', [prb.getState().getVars(), prb.getInput().getVars()], [wdot])

qddot.addVariable(cs.vcat([rddot, wdot]))
for i in range(0, nc):
    qddot.addVariable(cddot[i])
xdot = utils.double_integrator_with_floating_base(q.getVars(), qdot.getVars(), qddot.getVars(), base_velocity_reference_frame=cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
prb.setDynamics(xdot)
prb.setDt(T/ns)
transcription_method = rospy.get_param("transcription_method", 'multiple_shooting')  # can choose between 'multiple_shooting' and 'direct_collocation'
transcription_opts = dict(integrator='RK2') # integrator used by the multiple_shooting

# foot_frames parameters are used to retrieve initial position of the contacts given the initial pose of the robot.
# note: the order of the contacts state/control variable is the order in which these contacts are set in the param server 

foot_frames = rospy.get_param("foot_frames")
if len(foot_frames) == 0:
    print("foot_frames parameter is mandatory, exiting...")
    exit()
if(len(foot_frames) != nc):
    print(f"foot frames number should match number of contacts! {len(foot_frames)} != {nc}")
    exit()
print(f"foot_frames: {foot_frames}")

max_contact_force = rospy.get_param("max_contact_force", 1000.)
print(f"max_contact_force: {max_contact_force}")
max_contact_velocity = rospy.get_param("max_contact_velocity", 10.)
print(f"max_contact_velocity: {max_contact_velocity}")
i = 0
initial_foot_position = dict()
for frame in foot_frames:
    FK = kindyn.fk(frame)
    p = FK(q=joint_init)['ee_pos']
    print(f"{frame}: {p}")
    
    # storing initial foot_position and setting as initial bound
    initial_foot_position[i] = p
    c[i].setInitialGuess(p)
    c[i].setBounds(p, p, 0)

    # contacts initial velocity is 0
    cdot[i].setInitialGuess([0., 0., 0.])
    cdot[i].setBounds([0., 0., 0.], [0., 0., 0.])  # with 0 velocity

    # forces are between -max_max_contact_force and max_max_contact_force (unilaterality is added later)
    f[i].setBounds([-max_contact_force, -max_contact_force, -max_contact_force], [max_contact_force, max_contact_force, max_contact_force])

    i = i + 1

# initialize com state and com velocity
COM = kindyn.centerOfMass()
com = COM(q=joint_init)['com']
r.setInitialGuess(com)
rdot.setInitialGuess([0., 0., 0.])

# initialize base state and base angular velocity
o.setInitialGuess(joint_init[3:7])
w.setInitialGuess([0., 0., 0.])

# weights
r_tracking_gain = rospy.get_param("r_tracking_gain", 1e3)
orientation_tracking_gain = prb.createParameter('orientation_tracking_gain', 1)
orientation_tracking_gain.assign(0.)
rdot_tracking_gain = rospy.get_param("rdot_tracking_gain", 1e4)
w_tracking_gain = rospy.get_param("w_tracking_gain", 1e4)
rel_pos_gain = rospy.get_param("rel_position_gain", 1e4)
force_switch_weight = rospy.get_param("force_switch_weight", 1e2)
min_qddot_gain = rospy.get_param("min_qddot_gain", 1e0)
min_f_gain = rospy.get_param("min_f_gain", 1e-2)

# fixme: where do these come from?
d_initial_1 = -(initial_foot_position[0][0:2] - initial_foot_position[2][0:2])
d_initial_2 = -(initial_foot_position[1][0:2] - initial_foot_position[3][0:2])

# create contact reference and contact switch
c_ref = dict()
cdot_switch = dict()
for i in range(0, nc):
    c_ref[i] = prb.createParameter("c_ref" + str(i), 1)
    c_ref[i].assign(initial_foot_position[i][2], nodes=range(0, ns+1))
    cdot_switch[i] = prb.createParameter("cdot_switch" + str(i), 1)
    cdot_switch[i].assign(1., nodes=range(0, ns + 1))

# create constraints
r.setBounds(com, com, 0)
rdot.setBounds([0., 0., 0.], [0., 0., 0.], 0)
o.setBounds(joint_init[3:7], joint_init[3:7], 0)
w.setBounds([0., 0., 0.], [0., 0., 0.], 0)

# contact position constraints
if contact_model > 1:
    for i in range(1, contact_model):
        prb.createConstraint("relative_vel_left_" + str(i), cdot[0][0:2] - cdot[i][0:2])
    for i in range(contact_model + 1, 2 * contact_model):
        prb.createConstraint("relative_vel_right_" + str(i), cdot[contact_model][0:2] - cdot[i][0:2])

# friction cone constraints
for i, fi in f.items():
    mu = rospy.get_param("friction_cone_coefficient", 0.8)
    stanceR = np.identity(3, dtype=float)  # environment rotation wrt inertial frame
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(fi, mu, stanceR)
    #prb.createIntermediateConstraint(f"f{i}_friction_cone", fc, bounds=dict(lb=fc_lb, ub=fc_ub))

for i in range(0, nc):
    prb.createConstraint("cz_tracking" + str(i), c[i][2] - c_ref[i])
    prb.createConstraint("cdotxy_tracking" + str(i), cdot_switch[i] * cdot[i][0:2])

# create cost function terms
prb.createResidual("rz_tracking",   np.sqrt(r_tracking_gain)    * (r[2] - com[2]),                       nodes=range(1, ns+1))
prb.createResidual("o_tracking",    orientation_tracking_gain   * (o - joint_init[3:7]),                 nodes=range(1, ns+1))
prb.createResidual("rdot_tracking", np.sqrt(rdot_tracking_gain) * (rdot - rdot_ref),                     nodes=range(1, ns+1))
prb.createResidual("w_tracking",    np.sqrt(w_tracking_gain)    * (w - w_ref),                           nodes=range(1, ns+1))
prb.createResidual("rel_pos_y_1_4", np.sqrt(rel_pos_gain)       * (-c[0][1] + c[2][1] - d_initial_1[1]), nodes=range(1, ns+1))
prb.createResidual("rel_pos_x_1_4", np.sqrt(rel_pos_gain)       * (-c[0][0] + c[2][0] - d_initial_1[0]), nodes=range(1, ns+1))
prb.createResidual("rel_pos_y_3_6", np.sqrt(rel_pos_gain)       * (-c[1][1] + c[3][1] - d_initial_2[1]), nodes=range(1, ns+1))
prb.createResidual("rel_pos_x_3_6", np.sqrt(rel_pos_gain)       * (-c[1][0] + c[3][0] - d_initial_2[0]), nodes=range(1, ns+1))
prb.createResidual("min_qddot",     np.sqrt(min_qddot_gain)     * (qddot.getVars()),                     nodes=range(0, ns))
for i in range(0, nc):
    prb.createResidual("min_f" + str(i),         force_scaling * np.sqrt(min_f_gain) * f[i],  nodes=range(0, ns))
    prb.createResidual("f" + str(i) + "_active", force_scaling * np.sqrt(force_switch_weight)
                                                 * (1. - cdot_switch[i]) * f[i],              nodes=range(0, ns))

# create solver
max_iteration = rospy.get_param("max_iteration", 20)
print(f"max_iteration: {max_iteration}")

initial_state = np.array([0.,    -0.15, 0.88,
                          0.,     0.,   0.,   1.,
                          0.115,  0.,   0.,
                         -0.095,  0.,   0.,
                          0.115, -0.3,  0.,
                         -0.095, -0.3,  0.,
                          0.,     0.,   0.,
                          0.,     0.,   0.,
                          0.,     0.,   0.,
                          0.,     0.,   0.,
                          0.,     0.,   0.,
                          0.,     0.,   0.])
static_input = np.array([0., 0., 0., 0., 0., m * 9,81 / force_scaling / 4,
                         0., 0., 0., 0., 0., m * 9,81 / force_scaling / 4,
                         0., 0., 0., 0., 0., m * 9,81 / force_scaling / 4,
                         0., 0., 0., 0., 0., m * 9,81 / force_scaling / 4])

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
solver = ddp.DDPSolver(prb, opts=opts)

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
dae["x"] = cs.vertcat(prb.getState().getVars())
dae["ode"] = prb.getDynamics()
dae["p"] = cs.vertcat(prb.getInput().getVars())
dae["quad"] = 0.
simulation_euler_integrator = integrators.EULER(dae)

# Walking patter generator and scheduler
wpg = wpg.steps_phase(f, c, cdot, initial_foot_position[0][2].__float__(), c_ref, cdot_switch, ns, number_of_legs=2,
                      contact_model=contact_model, max_force=max_contact_force, max_velocity=max_contact_velocity)
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
        rdot_ref.assign(rdot_ref.getValues(nodes=j), nodes=j-1)
        w_ref.assign(w_ref.getValues(nodes=j), nodes=j-1)

    # assign new references based on user input
    if motion == "standing":
        alphaX, alphaY = 0.1, 0.1
    else:
        alphaX, alphaY = 1., 1.0

    if joy_msg is not None:
        rdot_ref.assign([alphaX * joy_msg.axes[1], alphaY * joy_msg.axes[0], 0.1 * joy_msg.axes[7]], nodes=ns)
        w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]], nodes=ns)
        orientation_tracking_gain.assign(cs.sqrt(1e5) if rotate else 0.)
    else:
        axis_x = keyboard.is_pressed('up') - keyboard.is_pressed('down')
        axis_y = keyboard.is_pressed('right') - keyboard.is_pressed('left')

        rdot_ref.assign([alphaX * axis_x, alphaY * axis_y, 0], nodes=ns)
        w_ref.assign([0, 0, 0], nodes=ns)
        orientation_tracking_gain.assign(0.)

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
    for i in range(0, nc):
        c0_hist['c' + str(i)] = solution['c' + str(i)][:, 0]

    t = rospy.Time().now()
    utilities.SRBDTfBroadcaster(solution['r'][:, 0], solution['o'][:, 0], c0_hist, t)
    for i in range(0, nc):
        viz.publishContactForce(t, force_scaling * solution['f' + str(i)][:, 0], 'c' + str(i))
        viz.publishPointTrj(solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
    viz.SRBDViewer(I, "SRB", t, nc)  # TODO: should we use w_R_b * I * w_R_b.T?
    viz.publishPointTrj(solution["r"], t, "SRB", "world")

    cc = dict()
    ff = dict()
    for i in range(0, nc):
        cc[i] = solution["c" + str(i)][:, 0]
        ff[i] = solution["f" + str(i)][:, 0]

    # simulation integration
    input = solution["u_opt"][:, 0]
    state = simulation_euler_integrator(state, input, prb.getDt())[0]
    #print(f"state:", solution["x_opt"])
    #print(f"input:", solution["u_opt"])
    rddot0 = RDDOT(input)
    wdot0 = WDOT(state, input)

    srbd_0 = kin_dyn.SRBD(m/force_scaling, I/force_scaling, ff, solution["r"][:, 0], rddot0, cc, solution["w"][:, 0], wdot0)
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
