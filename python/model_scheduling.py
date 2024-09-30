#!/usr/bin/env python
import logging

class model_params:
    def __init__(self, ns, T):
        self.ns = ns
        self.T = T

import time
from horizon.ros import utils as horizon_ros_utils
from ttictoc import tic,toc
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float32
import viz
import cartesio #todo: use bindings!
import numpy as np
import keyboard
import rospy
import prb as model_problem
import casadi as cs
import utilities
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
import ddp
from horizon.utils import mat_storer
import tf

horizon_ros_utils.roslaunch("srbd_horizon", "model_scheduling.launch")
time.sleep(3.)

full_params = model_params(ns=5, T=0.25)
srbd_params = model_params(ns=5, T=0.25)
lip_params  = model_params(ns=10, T=0.5)

rospy.init_node('model_scheduling_test', anonymous=True)
solution_time_pub = rospy.Publisher("solution_time", Float32, queue_size=10)
joint_state_publisher = rospy.Publisher("joint_states", JointState, queue_size=10)
rate = rospy.Rate(rospy.get_param("hz", 10)) # 10 Hz

full_model = model_problem.FullBodyProblem("full_model")
full_model.createFullBodyProblem(full_params.ns, full_params.T, include_transmission_forces=False)

srbd = model_problem.SRBDProblem("srbd")
srbd.createSRBDProblem(srbd_params.ns, srbd_params.T)

lip = model_problem.LIPProblem("srbd")
lip.createLIPProblem(lip_params.ns, lip_params.T)


sqp_opts = dict()
sqp_opts["gnsqp.max_iter"] = 1
sqp_opts['gnsqp.osqp.scaled_termination'] = False
sqp_opts['gnsqp.eps_regularization'] = 1e-2
sqp_opts['gnsqp.osqp.polish'] = False
sqp_opts['gnsqp.osqp.verbose'] = False

solver_sqp = ddp.SQPSolver(full_model.prb, qp_solver_plugin='osqp', opts=sqp_opts)
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
solver_sqp.setInitialGuess(full_model.getInitialGuess())
X, U = full_model.getStateInputMappingMatrices()
solver_sqp.setStateInputMapping(X, U)
P = full_model.getVarStateMappingMatrix()
solver_sqp.setVariableStateMapping(P)
O = full_model.getVarInputMappingMatrix()
solver_sqp.setVariableInputMapping(O)

variables_dict = {"q": full_model.q, "qdot": full_model.qdot, "qddot": full_model.qddot}
if full_model.include_transmission_forces:
    variables_dict["left_actuation_lambda"] = full_model.left_actuation_lambda
    variables_dict["right_actuation_lambda"] = full_model.right_actuation_lambda
for foot_frame in full_model.foot_frames:
    variables_dict["f_" + foot_frame] = full_model.f[foot_frame]


ddp_opts = dict()
ddp_opts["max_iters"] = 100
ddp_opts["alpha_converge_threshold"] = 1e-12
ddp_opts["beta"] = 1e-3
solver_srbd = ddp.DDPSolver(srbd.prb, opts=ddp_opts)
srbd_state = srbd.getInitialState()
srbd_x_warmstart = np.zeros((srbd_state.shape[0], srbd_params.ns + 1))
for i in range(0, srbd_params.ns+1):
    srbd_x_warmstart[:, i] = srbd_state
srbd_u_warmstart = np.zeros((srbd.getStaticInput().shape[0], srbd_params.ns))
for i in range(0, srbd_params.ns):
    srbd_u_warmstart[:, i] = srbd.getStaticInput()
solver_srbd.set_x_warmstart(srbd_x_warmstart)
solver_srbd.set_u_warmstart(srbd_u_warmstart)

solver_lip = ddp.DDPSolver(lip.prb, opts=ddp_opts)
lip_state = lip.getInitialState()
lip_x_warmstart = np.zeros((lip_state.shape[0], lip_params.ns + 1))
for i in range(0, lip_params.ns+1):
    lip_x_warmstart[:, i] = lip_state
lip_u_warmstart = np.zeros((lip.getStaticInput().shape[0], lip_params.ns))
for i in range(0, lip_params.ns):
    lip_u_warmstart[:, i] = lip.getStaticInput()
solver_lip.set_x_warmstart(lip_x_warmstart)
solver_lip.set_u_warmstart(lip_u_warmstart)



print(f"lip com initial state: {lip.getInitialState()[0:3]}")
print(f"srbd com initial state: {srbd.getInitialState()[0:3]}")
print(f"full com initial state: {full_model.kindyn.centerOfMass()(q=full_model.getInitialState()[0:full_model.nq])['com']}")

print(f"srbd orientation initial state: {srbd.getInitialState()[3:7]}")
print(f"full orientation initial state: {full_model.getInitialState()[3:7]}")

print(f"lip left_foot_upper initial state: {lip.getInitialState()[3:6]}")
print(f"srbd left_foot_upper initial state: {srbd.getInitialState()[7:10]}")
c_left_foot_upper = full_model.kindyn.fk("left_foot_upper")(q=full_model.getInitialState()[0:full_model.nq])['ee_pos']
print(f"full left_foot_upper initial state: {c_left_foot_upper}")

print(f"lip left_foot_lower initial state: {lip.getInitialState()[6:9]}")
print(f"srbd left_foot_lower initial state: {srbd.getInitialState()[10:13]}")
c_left_foot_lower = full_model.kindyn.fk("left_foot_lower")(q=full_model.getInitialState()[0:full_model.nq])['ee_pos']
print(f"full left_foot_lower initial state: {c_left_foot_lower}")

print(f"lip right_foot_upper initial state: {lip.getInitialState()[9:12]}")
print(f"srbd right_foot_upper initial state: {srbd.getInitialState()[13:16]}")
c_right_foot_upper = full_model.kindyn.fk("right_foot_upper")(q=full_model.getInitialState()[0:full_model.nq])['ee_pos']
print(f"full right_foot_upper initial state: {c_right_foot_upper}")

print(f"lip right_foot_lower initial state: {lip.getInitialState()[12:15]}")
print(f"srbd right_foot_lower initial state: {srbd.getInitialState()[16:19]}")
c_right_foot_lower = full_model.kindyn.fk("right_foot_lower")(q=full_model.getInitialState()[0:full_model.nq])['ee_pos']
print(f"full left_foot_lower initial state: {c_right_foot_lower}")


meta_solver = ddp.MetaSolver(full_model.prb, None)

com = full_model.kindyn.centerOfMass()(q=full_model.q)['com']
vcom = full_model.kindyn.centerOfMass()(q=full_model.q, v=full_model.qdot)['vcom']
c_left_foot_upper = full_model.kindyn.fk("left_foot_upper")(q=full_model.q)['ee_pos']
c_left_foot_lower = full_model.kindyn.fk("left_foot_lower")(q=full_model.q)['ee_pos']
c_right_foot_upper = full_model.kindyn.fk("right_foot_upper")(q=full_model.q)['ee_pos']
c_right_foot_lower = full_model.kindyn.fk("right_foot_lower")(q=full_model.q)['ee_pos']
vc_left_foot_upper = full_model.kindyn.frameVelocity("left_foot_upper", cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)(q=full_model.q, qdot=full_model.qdot)['ee_vel_linear']
vc_left_foot_lower = full_model.kindyn.frameVelocity("left_foot_lower", cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)(q=full_model.q, qdot=full_model.qdot)['ee_vel_linear']
vc_right_foot_upper = full_model.kindyn.frameVelocity("right_foot_upper", cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)(q=full_model.q, qdot=full_model.qdot)['ee_vel_linear']
vc_right_foot_lower = full_model.kindyn.frameVelocity("right_foot_lower", cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)(q=full_model.q, qdot=full_model.qdot)['ee_vel_linear']
full_to_srbd_function = cs.Function("full_to_srbd", [full_model.prb.getState().getVars()],
                                    [cs.vcat([com, # [0:3]
                                     full_model.q[3:7], # [3:7] base orientation
                                     c_left_foot_upper, # [7:10 ]
                                     c_left_foot_lower, # [10:13]
                                     c_right_foot_upper, # [13:16]
                                     c_right_foot_lower, # [16:19]
                                     vcom,
                                     full_model.qdot[3:6],  # base angular velocity
                                     vc_left_foot_upper,
                                     vc_left_foot_lower,
                                     vc_right_foot_upper,
                                     vc_right_foot_lower])])
meta_solver.addSQP(solver_sqp, full_to_srbd_function)

srbd_to_lip_function = cs.Function("srbd_to_lip", [srbd.prb.getState().getVars()],
                                   [cs.vcat([srbd.prb.getState().getVars()[0:3], # com
                                             srbd.prb.getState().getVars()[7:19], # contacts
                                             srbd.prb.getState().getVars()[19:22], # vcom
                                             srbd.prb.getState().getVars()[25:37]])]) # vcontacts
meta_solver.add(solver_srbd, srbd_to_lip_function)

foo_mapping_function = cs.Function("foo", [lip.prb.getState().getVars()], [cs.DM.zeros(1, 1)])
meta_solver.add(solver_lip, foo_mapping_function)

#meta_solver.setInitialState(full_model.getInitialState()) # this is not used when first solver is SQP, x0 is set through constraints!

# IMPORTANT: update bounds and constraint of SQP todo: can we do it internally to the solver?
solver_sqp.updateBounds()
solver_sqp.updateConstraints()

meta_solver.solve()

#solution = meta_solver.getSolutionDict()
solution = meta_solver.getSolutionModel(0)
srbd_solution = meta_solver.getSolutionModel(1)
lip_solution = meta_solver.getSolutionModel(2)

solution['q'] = utilities.normalize_quaternion_part_horizon(solution['q'], full_model.ns)

joint_state_msg = JointState()
joint_state_msg.name = full_model.kindyn.joint_names()[2:]

while not rospy.is_shutdown():
    t = rospy.Time.now()

    lip_input = lip_solution["u_opt"][:, 0]
    lip_state = lip_solution["x_opt"][:, 0]
    utilities.ZMPTfBroadcaster(lip_solution['z'][:, 0], t)
    rddot0 = lip.RDDOT(lip_state, lip_input, solver_lip.get_params_value(0))
    fzmp = lip.m * (np.array([0., 0., 9.81]) + rddot0)
    viz.publishContactForce(t, fzmp, 'ZMP')
    viz.publishPointTrj(lip_solution["r"], t, name="COM", frame="world", color=[1., 1., 0.], namespace="LIP")
    viz.publishPointTrj(lip_solution["z"], t, name="ZMP", frame="world", color=[0., 1., 1.], namespace="LIP")

    c0_hist = dict()
    for i in range(0, srbd.nc):
        c0_hist['c' + str(i)] = srbd_solution['c' + str(i)][:, 0]
    utilities.SRBDTfBroadcaster(srbd_solution['r'][:, 0], srbd_solution['o'][:, 0], c0_hist, t)
    for i in range(0, srbd.nc):
        viz.publishContactForce(t, srbd.force_scaling * srbd_solution['f' + str(i)][:, 0], 'c' + str(i))
        viz.publishPointTrj(srbd_solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
    viz.SRBDViewer(srbd.I, "SRB", t, srbd.nc)  # TODO: should we use w_R_b * I * w_R_b.T?
    viz.publishPointTrj(srbd_solution["r"], t, "SRB", "world")

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
        c[foot_frame] = np.zeros((3, full_model.ns + 1))
        for i in range(0, full_model.ns + 1):
            c[foot_frame][:, i] = C(q=solution['q'][:, i])['ee_pos'].toarray().flatten()
    for i in range(0, full_model.nc):
        viz.publishContactForce(t, solution['f_' + full_model.foot_frames[i]][:, 0], frame=full_model.foot_frames[i],
                                topic='fm_fc' + str(i))
        viz.publishPointTrj(c[full_model.foot_frames[i]], t, 'fm_c' + str(i), "world", color=[0., 0., 1.])

# while not rospy.is_shutdown():
#     """
#     Automatically set initial guess from solution to variables in variables_dict
#     """
#     mat_storer.setInitialGuess(variables_dict, solution)
#     solver_sqp.setInitialGuess(full_model.getInitialGuess())
#
#     #open loop
#     full_model.q.setBounds(solution['q'][:, 1], solution['q'][:, 1], 0)
#     full_model.qdot.setBounds(solution['qdot'][:, 1], solution['qdot'][:, 1], 0)
#
#     solver_sqp.updateBounds()
#     solver_sqp.updateConstraints()
#
#     # solve
#     tic()
#     meta_solver.solve()
#     solution_time_pub.publish(toc())
#
#     solution = meta_solver.getSolutionDict()
#     srbd_solution = meta_solver.getSolutionModel(1)
#     lip_solution = meta_solver.getSolutionModel(2)
#
#     solution['q'] = utilities.normalize_quaternion_part_horizon(solution['q'], full_model.ns)
#
#
#
#

#

#
#     # publish contact forces and contact points
#     c = dict()
#     for foot_frame in full_model.foot_frames:
#         C = full_model.kindyn.fk(foot_frame)
#         c[foot_frame] = np.zeros((3, full_model.ns + 1))
#         for i in range(0, full_model.ns + 1):
#             c[foot_frame][:, i] = C(q=solution['q'][:, i])['ee_pos'].toarray().flatten()
#     for i in range(0, full_model.nc):
#         viz.publishContactForce(t, solution['f_' + full_model.foot_frames[i]][:, 0], frame=full_model.foot_frames[i],
#                                 topic='fc' + str(i))
#         viz.publishPointTrj(c[full_model.foot_frames[i]], t, 'c' + str(i), "world", color=[0., 0., 1.])
#
#     # publish center of mass
#     COM = full_model.kindyn.centerOfMass()
#     com = np.zeros((3, full_model.ns + 1))
#     for i in range(0, full_model.ns + 1):
#         com[:, i] = COM(q=solution['q'][:, i])['com'].toarray().flatten()
#     viz.publishPointTrj(com, t, "SRB", "world")
#
#     rate.sleep()















