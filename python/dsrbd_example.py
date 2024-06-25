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

SOLVER = lambda: 'ipopt'

def joy_cb(msg):
    global joy_msg
    joy_msg = msg


horizon_ros_utils.roslaunch("srbd_horizon", "SRBD_kangaroo_line_feet.launch")
time.sleep(3.)

"""
Creates HORIZON problem. 
These parameters can not be tuned at the moment.
"""
ns = 20
prb = problem.Problem(ns, casadi_type=cs.SX)
T = 1.


urdf = rospy.get_param("robot_description", "")
if urdf == "":
    print("robot_description not loaded in param server!")
    exit()

kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

"""
Creates problem STATE variables
"""
""" CoM Position """
r = prb.createStateVariable("r", 3)
""" Base orientation (quaternion) """
o = prb.createStateVariable("o", 4)

""" Variable to collect all position states """
q = variables.Aggregate()
q.addVariable(r)
q.addVariable(o)

""" Contacts position """
contact_model = rospy.get_param("contact_model", 4)
print(f"contact_model: {contact_model}")

number_of_legs = rospy.get_param("number_of_legs", 2)
print(f"number_of_legs: {number_of_legs}")

nc = number_of_legs * contact_model
print(f"nc: {nc}")

c = dict()
for i in range(0, nc):
    c[i] = prb.createStateVariable("c" + str(i), 3) # Contact i position
    q.addVariable(c[i])

""" CoM Velocity and paramter to handle references """
rdot = prb.createStateVariable("rdot", 3) # CoM vel
rdot_ref = prb.createParameter('rdot_ref', 3)
rdot_ref.assign([0., 0., 0.], nodes=range(1, ns+1))

""" Base angular Velocity and parameter to handle references """
w = prb.createStateVariable("w", 3) # base vel
w_ref = prb.createParameter('w_ref', 3)
w_ref.assign([0., 0., 0.], nodes=range(1, ns+1))

""" Variable to collect all velocity states """
qdot = variables.Aggregate()
qdot.addVariable(rdot)
qdot.addVariable(w)

""" Contacts velocity """
cdot = dict()
for i in range(0, nc):
    cdot[i] = prb.createStateVariable("cdot" + str(i), 3)  # Contact i vel
    qdot.addVariable(cdot[i])

"""
Contacts acceleration and forces
"""
""" Variable to collect all acceleration controls """
qddot = variables.Aggregate()

cddot = dict()
f = dict()
for i in range(0, nc):
    cddot[i] = prb.createInputVariable("cddot" + str(i), 3) # Contact i acc
    f[i] = prb.createInputVariable("f" + str(i), 3) # Contact i forces

"""
Formulate discrete time dynamics using multiple_shooting and RK2 integrator
joint_init is used to initialize the urdf model and retrieve information such as: CoM, Inertia, atc... 
at the nominal configuration given by joint_init
"""
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
rddot, wdot = kin_dyn.fSRBD(m, w_R_b * I * w_R_b.T, f, r, c, w)

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

#todo: ipopt
#if transcription_method == 'direct_collocation':
#    transcription_opts = dict()
#th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)


"""
foot_frames parameters are used to retrieve initial position of the contacts given the initial pose of the robot.
note: the order of the contacts state/control variable is the order in which these contacts are set in the param server 
"""
foot_frames = rospy.get_param("foot_frames")
if len(foot_frames) == 0:
    print("foot_frames parameter is mandatory, exiting...")
    exit()
if(len(foot_frames) != nc):
    print(f"foot frames number shopuld match with number of contacts! {len(foot_frames)} != {nc}")
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
    """
    Storing initial foot_position and setting as initial bound
    """
    initial_foot_position[i] = p
    c[i].setInitialGuess(p)
    c[i].setBounds(p, p, 0)

    """
    Contacts initial velocity is 0
    """
    cdot[i].setInitialGuess([0., 0., 0.])
    cdot[i].setBounds([0., 0., 0.], [0., 0., 0.])  # with 0 velocity

    """
    Forces are between -max_max_contact_force and max_max_contact_force (unilaterality is added later)
    """
    f[i].setBounds([-max_contact_force, -max_contact_force, -max_contact_force], [max_contact_force, max_contact_force, max_contact_force])

    i = i + 1

"""
Initialize com state and com velocity
"""
COM = kindyn.centerOfMass()
com = COM(q=joint_init)['com']
print(f"com: {com}")
r.setInitialGuess(com)
r.setBounds(com, com, 0)
rdot.setBounds([0., 0., 0.], [0., 0., 0.], 0)
rdot.setInitialGuess([0., 0., 0.])

"""
Initialize base state and base angular velocity
"""
print(f"base orientation: {joint_init[3:7]}")
o.setInitialGuess(joint_init[3:7])
o.setBounds(joint_init[3:7], joint_init[3:7], 0)
w.setInitialGuess([0., 0., 0.])
w.setBounds([0., 0., 0.], [0., 0., 0.], 0)

"""
Set up some therms of the COST FUNCTION
"""
"""
rz_tracking is used to keep the com height around the initial value
"""
rz_tracking_gain = rospy.get_param("rz_tracking_gain", 2e3)
print(f"rz_tracking_gain: {rz_tracking_gain}")
prb.createResidual("rz_tracking", np.sqrt(rz_tracking_gain) *  (r[2] - com[2]), nodes=range(1, ns+1))

"""
o_tracking is used to keep the base orientation at identity, its gain is initialize at 0 and set to non-0 only when a button is pressed
"""
Wo = prb.createParameter('Wo', 1)
Wo.assign(0.)
prb.createResidual("o_tracking", Wo * (o - joint_init[3:7]), nodes=range(1, ns+1))

"""
rdot_tracking is used to track a desired velocity of the CoM
"""
rdot_tracking_gain = rospy.get_param("rdot_tracking_gain", 1e4)
print(f"rdot_tracking_gain: {rdot_tracking_gain}")
prb.createResidual("rdot_tracking", np.sqrt(rdot_tracking_gain) * (rdot - rdot_ref), nodes=range(1, ns+1))

"""
w_tracking is used to track a desired angular velocity of the base
"""
w_tracking_gain = rospy.get_param("w_tracking_gain", 1e4)
print(f"w_tracking_gain: {w_tracking_gain}")
prb.createResidual("w_tracking", np.sqrt(w_tracking_gain) * (w - w_ref), nodes=range(1, ns+1))

"""
min_qddot is to minimize the acceleration control effort
"""
min_qddot_gain = rospy.get_param("min_qddot_gain", 1e0)
print(f"min_qddot_gain: {min_qddot_gain}")
prb.createResidual("min_qddot", np.sqrt(min_qddot_gain) * (qddot.getVars()), nodes=list(range(0, ns)))

"""
Set up som CONSTRAINTS
"""
"""
These are the relative distance in y between the feet. Initial configuration of contacts is taken as minimum distance in Y! 
TODO: when feet will rotates, also these constraint has to rotate!
TODO: what happen for only 4 contacts???
"""
max_clearance_x = rospy.get_param("max_clearance_x", 0.5)
print(f"max_clearance_x: {max_clearance_x}")
max_clearance_y = rospy.get_param("max_clearance_y", 0.5)
print(f"max_clearance_y: {max_clearance_y}")

fpi = []
for l in range(0, number_of_legs):
    if contact_model == 1:
        fpi.append(l)
    else:
        fpi.append(l * contact_model)
        fpi.append(l * contact_model + contact_model - 1)

d_initial_1 = -(initial_foot_position[fpi[0]][0:2] - initial_foot_position[fpi[2]][0:2])
relative_pos_y_1_4 = prb.createResidual("relative_pos_y_1_4", 1e2 * (-c[fpi[0]][1] + c[fpi[2]][1] - d_initial_1[1]))
relative_pos_x_1_4 = prb.createResidual("relative_pos_x_1_4", 1e2 * (-c[fpi[0]][0] + c[fpi[2]][0] - d_initial_1[0]))
d_initial_2 = -(initial_foot_position[fpi[1]][0:2] - initial_foot_position[fpi[3]][0:2])
relative_pos_y_3_6 = prb.createResidual("relative_pos_y_3_6", 1e2 * (-c[fpi[1]][1] + c[fpi[3]][1] - d_initial_2[1]))
relative_pos_x_3_6 = prb.createResidual("relative_pos_x_3_6", 1e2 * (-c[fpi[1]][0] + c[fpi[3]][0] - d_initial_2[0]))

min_f_gain = rospy.get_param("min_f_gain", 1e-2)
print(f"min_f_gain: {min_f_gain}")
c_ref = dict()
for i in range(0, nc):
    """
    min_f try to minimze the contact forces (can be seen as distribute equally the contact forces)
    """
    prb.createResidual("min_f" + str(i), np.sqrt(min_f_gain) * f[i], nodes=list(range(0, ns)))

    """
    cz_tracking is used to track the z reference for the feet: notice that is a constraint
    """
    c_ref[i] = prb.createParameter("c_ref" + str(i), 1)
    c_ref[i].assign(initial_foot_position[i][2], nodes=range(0, ns+1))
    prb.createConstraint("cz_tracking" + str(i), c[i][2] - c_ref[i])

"""
Friction cones and force unilaterality constraint
TODO: for now flat terrain is assumed (StanceR needs tio be used more or less everywhere for contacts)
"""
mu = rospy.get_param("friction_cone_coefficient", 0.8)
print(f"mu: {mu}")
for i, fi in f.items():
    # FRICTION CONE
    StanceR = np.identity(3, dtype=float)  # environment rotation wrt inertial frame
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(fi, mu, StanceR)
    prb.createIntermediateConstraint(f"f{i}_friction_cone", fc, bounds=dict(lb=fc_lb, ub=fc_ub))

"""
This constraint is used to keep points which belong to the same contacts together
note: needs as well to be rotated in future to consider w x p
TODO: use also number_of_legs
"""
if contact_model > 1:
    for i in range(1, contact_model):
        prb.createConstraint("relative_vel_left_" + str(i), cdot[0][0:2] - cdot[i][0:2])
    for i in range(contact_model + 1, 2 * contact_model):
        prb.createConstraint("relative_vel_right_" + str(i), cdot[contact_model][0:2] - cdot[i][0:2])

"""
Create solver
"""
max_iteration = rospy.get_param("max_iteration", 20)
print(f"max_iteration: {max_iteration}")

i_opts = solver_options.ipopt_offline_solver_options()
if SOLVER() == 'gnsqp':
    i_opts = solver_options.sqp_offline_solver_options(ns)

solver_offline = solver.Solver.make_solver(SOLVER(), prb, i_opts)
# solver_offline.set_iteration_callback()

solver_offline.solve()
solution = solver_offline.getSolutionDict()
state = solution["x_opt"][:,0]



"""
Dictionary to store variables used for warm-start
"""
variables_dict = {"r": r, "rdot": rdot,
                  "o": o, "w": w}
for i in range(0, nc):
    variables_dict["c" + str(i)] = c[i]
    variables_dict["cdot" + str(i)] = cdot[i]
    variables_dict["cddot" + str(i)] = cddot[i]
    variables_dict["f" + str(i)] = f[i]

rospy.init_node('srbd_mpc_test', anonymous=True)

hz = rospy.get_param("hz", 10)
print(f"hz: {hz}")
rate = rospy.Rate(hz)  # 10hz
rospy.Subscriber('/joy', Joy, joy_cb)
global joy_msg
joy_msg = None

solution_time_pub = rospy.Publisher("solution_time", Float32, queue_size=10)
srbd_pub = rospy.Publisher("srbd_constraint", WrenchStamped, queue_size=10)
srbd_msg = WrenchStamped()

"""
online_solver
"""
opts = solver_options.ipopt_online_solver_options(max_iteration)
if SOLVER() == 'gnsqp':
    opts = solver_options.sqp_online_solver_options(max_iterations=1)

#solver = solver.Solver.make_solver(SOLVER(), prb, opts)
import ddp
opts = dict()
opts["max_iters"] = 20
solver = ddp.DDPSolver(prb, opts=opts)
solver.set_u_warmstart(solution["u_opt"])

# solver.set_iteration_callback()

#define discrete dynamics
dae = dict()
dae["x"] = cs.vertcat(prb.getState().getVars())
dae["ode"] = prb.getDynamics()
dae["p"] = cs.vertcat(prb.getInput().getVars())
dae["quad"] = 0.
simulation_euler_integrator = integrators.EULER(dae)

"""
Walking patter generator and scheduler
"""
wpg = wpg.steps_phase(f, c, cdot, initial_foot_position[0][2].__float__(), c_ref, ns, number_of_legs=number_of_legs,
                      contact_model=contact_model, max_force=max_contact_force, max_velocity=max_contact_velocity)
ci = cartesio.cartesIO(["left_sole_link", "right_sole_link"])
while not rospy.is_shutdown():
    """
    Automatically set initial guess from solution to variables in variables_dict
    """
    mat_storer.setInitialGuess(variables_dict, solution)
    # open loop
    r.setBounds(solution['r'][:, 1], solution['r'][:, 1], 0)
    rdot.setBounds(solution['rdot'][:, 1], solution['rdot'][:, 1], 0)
    o.setBounds(solution['o'][:, 1], solution['o'][:, 1], 0)
    w.setBounds(solution['w'][:, 1], solution['w'][:, 1], 0)
    for i in range(0, nc):
        c[i].setBounds(solution['c' + str(i)][:, 1], solution['c' + str(i)][:, 1], 0)
        cdot[i].setBounds(solution['cdot' + str(i)][:, 1], solution['cdot' + str(i)][:, 1], 0)

    solver.setInitialState(state)

    motion = "standing"
    if joy_msg is not None:
        if joy_msg.buttons[4]:
            motion = "walking"
        if joy_msg.buttons[5]:
            motion = "jumping"
    else:
        if keyboard.is_pressed('ctrl'):
            motion = "walking"
        if keyboard.is_pressed('space'):
            motion = "jumping"

    if motion == "standing":
        alphaX, alphaY = 0.1, 0.1
    else:
        alphaX, alphaY = 0.4, 0.3

    if joy_msg is not None:
        rdot_ref.assign([alphaX * joy_msg.axes[1], alphaY * joy_msg.axes[0], 0.1 * joy_msg.axes[7]],
                        nodes=range(1, ns + 1))  # com velocities
        w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]],
                     nodes=range(1, ns + 1))  # base angular velocities
        if (joy_msg.buttons[3]):
            Wo.assign(cs.sqrt(1e5))
        else:
            Wo.assign(0.)
    else:
        axis_x = keyboard.is_pressed('up') - keyboard.is_pressed('down')
        axis_y = keyboard.is_pressed('right') - keyboard.is_pressed('left')

        rdot_ref.assign([alphaX * axis_x, alphaY * axis_y, 0], nodes=range(1, ns + 1))  # com velocities
        w_ref.assign([0, 0, 0], nodes=range(1, ns + 1))  # base angular velocities
        Wo.assign(0.)

    if motion == "walking":
        wpg.set("step")
    elif motion == "jumping":
        wpg.set("jump")
    else:
        wpg.set("standing")

    tic()
    solver.solve()
    #print("r: ", solution["r"])
    #print("rdot: ", solution["rdot"])
    #print("w: ", solution["w"])
    #exit()


    solution_time_pub.publish(toc())
    solution = solver.getSolutionDict()

    c0_hist = dict()
    for i in range(0, nc):
        c0_hist['c' + str(i)] = solution['c' + str(i)][:, 0]

    t = rospy.Time().now()
    utilities.SRBDTfBroadcaster(solution['r'][:, 0], solution['o'][:, 0], c0_hist, t)
    for i in range(0, nc):
        viz.publishContactForce(t, solution['f' + str(i)][:, 0], 'c' + str(i))
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
    rddot0 = RDDOT(input)
    wdot0 = WDOT(state, input)

    srbd_0 = kin_dyn.SRBD(m, I, ff, solution["r"][:, 0], rddot0, cc, solution["w"][:, 0], wdot0)
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
