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
from sensor_msgs.msg import Joy
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

horizon_ros_utils.roslaunch("srbd_horizon", "model_scheduling.launch")
time.sleep(3.)

full_params = model_params(ns=5, T=0.25)
srbd_params = model_params(ns=5, T=0.25)
lip_params  = model_params(ns=10, T=0.5)

full_model = model_problem.FullBodyProblem("full_model")
full_model.createFullBodyProblem(full_params.ns, full_params.T)

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
full_model.left_actuation_lambda.setInitialGuess(full_model.getStaticInput()[full_model.nv+i*3+3:full_model.nv+i*3+3+2])
full_model.right_actuation_lambda.setInitialGuess(full_model.getStaticInput()[full_model.nv+i*3+3+2:])
solver_sqp.setInitialGuess(full_model.getInitialGuess())
X, U = full_model.getStateInputMappingMatrices()
solver_sqp.setStateInputMapping(X, U)
P = full_model.getVarInputMappingMatrix()
solver_sqp.setVariableStateMapping(P)


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
                                    [cs.vcat([com,
                                     full_model.q[3:7], # base orientation
                                     c_left_foot_upper,
                                     c_left_foot_lower,
                                     c_right_foot_upper,
                                     c_right_foot_lower,
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
#solver_sqp.solve()
meta_solver.solve()

solution = meta_solver.getSolutionDict()
srbd_solution = meta_solver.getSolutionModel(1)
lip_solution = meta_solver.getSolutionModel(2)
print(srbd_solution)







