#!/usr/bin/env python
import logging

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
import prb as srbd_problem
import casadi as cs
import utilities
from horizon.utils import utils, kin_dyn


class steps_phase:
    def __init__(self, f, c, cdot, c_init_z, c_ref, w_ref, orientation_tracking_gain, cdot_switch, nodes,
                 number_of_legs, contact_model):
        self.f = f
        self.c = c
        self.cdot = cdot
        self.c_ref = c_ref
        self.cdot_switch = cdot_switch
        self.w_ref = w_ref
        self.orientation_tracking_gain = orientation_tracking_gain

        self.number_of_legs = number_of_legs
        self.contact_model = contact_model

        self.nodes = nodes
        self.step_counter = 0

        self.step_duration = 0.5
        self.dt = 0.05
        self.ss_share = 0.8
        self.ds_share = 0.2
        self.step_nodes = int(self.step_duration / self.dt)

        # generate step cycle
        ss_duration = int(self.ss_share * self.step_nodes)
        ds_duration = int(self.ds_share * self.step_nodes)
        sin = 0.1 * np.sin(np.linspace(0, np.pi, ))

        # left step cycle
        self.l_cycle = []
        self.l_cdot_switch = []
        for k in range(0, ds_duration):
            self.l_cycle.append(c_init_z)
            self.l_cdot_switch.append(1.)
        for k in range(0, ss_duration):
            self.l_cycle.append(c_init_z + sin[k + 1])
            self.l_cdot_switch.append(0.)
        for k in range(0, ds_duration):
            self.l_cycle.append(c_init_z)
            self.l_cdot_switch.append(1.)
        for k in range(0, ss_duration):
            self.l_cycle.append(c_init_z)
            self.l_cdot_switch.append(1.)
        self.l_cycle.append(c_init_z)
        self.l_cdot_switch.append(1.)

        # right step cycle
        self.r_cycle = []
        self.r_cdot_switch = []
        for k in range(0, ds_duration):
            self.r_cycle.append(c_init_z)
            self.r_cdot_switch.append(1.)
        for k in range(0, ss_duration):
            self.r_cycle.append(c_init_z)
            self.r_cdot_switch.append(1.)
        for k in range(0, ds_duration):
            self.r_cycle.append(c_init_z)
            self.r_cdot_switch.append(1.)
        for k in range(0, ss_duration):
            self.r_cycle.append(c_init_z + sin[k + 1])
            self.r_cdot_switch.append(0.)
        self.r_cycle.append(c_init_z)
        self.r_cdot_switch.append(1.)

        self.action = ""

    def set(self, action):

        self.action = action
        ref_id = self.step_counter % (2 * self.step_nodes)


        # fill last node of the contact plan
        if self.action == "step":
            self.w_ref.assign([0, 0., 0.], nodes=self.nodes)
            self.orientation_tracking_gain.assign(1e2, nodes=self.nodes)
            for i in range(0, self.contact_model):
                self.cdot_switch[i].assign(self.l_cdot_switch[ref_id], nodes=self.nodes)
                self.c_ref[i].assign(self.l_cycle[ref_id], nodes=self.nodes)
            for i in range(self.contact_model, self.contact_model * self.number_of_legs):
                self.cdot_switch[i].assign(self.r_cdot_switch[ref_id], nodes=self.nodes)
                self.c_ref[i].assign(self.r_cycle[ref_id], nodes=self.nodes)
        elif self.action == "jump":
            self.w_ref.assign([0, 0., 0.], nodes=self.nodes)
            self.orientation_tracking_gain.assign(0., nodes=self.nodes)
            for i in range(0, len(self.c)):
                self.cdot_switch[i].assign(0., nodes=self.nodes)
        else:  # stance
            self.w_ref.assign([0, 0., 0.], nodes=self.nodes)
            self.orientation_tracking_gain.assign(1e2, nodes=self.nodes)
            for i in range(0, len(self.c)):
                self.cdot_switch[i].assign(1., nodes=self.nodes)
                self.c_ref[i].assign(0., nodes=self.nodes)

        self.step_counter += 1


def joy_cb(msg):
    global joy_msg
    joy_msg = msg


horizon_ros_utils.roslaunch("srbd_horizon", "SRBD_kangaroo_line_feet.launch")
time.sleep(3.)

# creates HORIZON problem, these parameters can not be tuned at the moment
ns_srbd = 10
ns_lip = 10
T_srbd = 0.5
T_lip = 0.5

srbd = srbd_problem.SRBDProblem()
srbd.createSRBDProblem(ns_srbd, T_srbd)
lip = srbd_problem.LIPProblem()
lip.createLIPProblem(ns_lip, T_lip)

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
solver_srbd = ddp.DDPSolver(srbd.prb, opts=opts)
solver_lip = ddp.DDPSolver(lip.prb, opts=opts)

# set initial state and warmstart ddp
lip_state = lip.getInitialState()
lip_x_warmstart = np.zeros((lip_state.shape[0], ns_lip + 1))
for i in range(0, ns_lip+1):
    lip_x_warmstart[:, i] = lip_state
lip_u_warmstart = np.zeros((lip.getStaticInput().shape[0], ns_lip))
for i in range(0, ns_lip):
    lip_u_warmstart[:, i] = lip.getStaticInput()

# set initial state and warmstart ddp
srbd_state = srbd.getInitialState()
srbd_x_warmstart = np.zeros((srbd_state.shape[0], ns_srbd + 1))
for i in range(0, ns_srbd+1):
    srbd_x_warmstart[:, i] = srbd_state
srbd_u_warmstart = np.zeros((srbd.getStaticInput().shape[0], ns_srbd))
for i in range(0, ns_srbd):
    srbd_u_warmstart[:, i] = srbd.getStaticInput()

solver_srbd.set_x_warmstart(srbd_x_warmstart)
solver_srbd.set_u_warmstart(srbd_u_warmstart)
solver_lip.set_x_warmstart(lip_x_warmstart)
solver_lip.set_u_warmstart(lip_u_warmstart)


#define discrete dynamics
dae = dict()
dae["x"] = cs.vertcat(srbd.prb.getState().getVars())
dae["ode"] = srbd.prb.getDynamics()
dae["p"] = cs.vertcat(srbd.prb.getInput().getVars())
dae["quad"] = 0.
srbd_euler_integrator = solver_srbd.get_f(0)


# Walking patter generator and scheduler
ci = cartesio.cartesIO(["left_sole_link", "right_sole_link"])

meta_solver = ddp.MetaSolver(srbd.prb, None)

#print(srbd.prb.getState().getVars())
#print(lip.prb.getState().getVars())

model_mapping_function = cs.Function("srbd_to_lip", [srbd.prb.getState().getVars()], [cs.vcat([srbd.prb.getState().getVars()[0:3], srbd.prb.getState().getVars()[7:19], srbd.prb.getState().getVars()[19:22], srbd.prb.getState().getVars()[25:37]])])
meta_solver.add(solver_srbd, model_mapping_function)
foo_mapping_function = cs.Function("foo", [lip.prb.getState().getVars()], [cs.DM.zeros(1, 1)])
meta_solver.add(solver_lip, foo_mapping_function)

wpg = steps_phase(srbd.f, lip.c, lip.cdot, lip.initial_foot_position[0][2].__float__(), lip.c_ref, srbd.w_ref, srbd.orientation_tracking_gain, lip.cdot_switch, ns_lip, number_of_legs=2,
                      contact_model=lip.contact_model)


while not rospy.is_shutdown():

    meta_solver.setInitialState(srbd_state)

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
    for j in range(1, ns_srbd):
        srbd.rdot_ref.assign(srbd.rdot_ref.getValues(nodes=j), nodes=j - 1)
        srbd.w_ref.assign(srbd.w_ref.getValues(nodes=j), nodes=j - 1)
        srbd.oref.assign(srbd.oref.getValues(nodes=j), nodes=j - 1)
        for i in range(0, srbd.nc):
            srbd.cdot_switch[i].assign(srbd.cdot_switch[i].getValues(nodes=j), nodes=j - 1)
            srbd.c_ref[i].assign(srbd.c_ref[i].getValues(nodes=j), nodes=j - 1)

    srbd.rdot_ref.assign(lip.rdot_ref.getValues(nodes=0), nodes=ns_srbd-1)
    for i in range(0, srbd.nc):
        srbd.cdot_switch[i].assign(lip.cdot_switch[i].getValues(nodes=0), nodes=ns_srbd-1)
        srbd.c_ref[i].assign(lip.c_ref[i].getValues(nodes=0), nodes=ns_srbd-1)
    #w_ref??
    #oref??

    for j in range(1, ns_lip + 1):
        lip.rdot_ref.assign(lip.rdot_ref.getValues(nodes=j), nodes=j-1)
        lip.eta2_p.assign(lip.eta2_p.getValues(nodes=j), nodes=j-1)
        for i in range(0, lip.nc):
            lip.cdot_switch[i].assign(lip.cdot_switch[i].getValues(nodes=j), nodes=j - 1)
            lip.c_ref[i].assign(lip.c_ref[i].getValues(nodes=j), nodes=j - 1)




    if lip.cdot_switch[0].getValues(ns_lip) == 0 and lip.cdot_switch[1].getValues(ns_lip) == 0 and lip.cdot_switch[2].getValues(ns_lip) == 0 and lip.cdot_switch[3].getValues(ns_lip) == 0:
        lip.eta2_p.assign(0., nodes=ns_lip)
    else:
        lip.eta2_p.assign(lip.eta2, nodes=ns_lip)



    # assign new references based on user input
    if motion == "standing":
        alphaX, alphaY = 0.1, 0.1
    else:
        alphaX, alphaY = 0.5, 0.5

    if joy_msg is not None:
        lip.rdot_ref.assign([alphaX * joy_msg.axes[1], alphaY * joy_msg.axes[0], 0.1 * joy_msg.axes[7]], nodes=ns_lip)
        #srbd.rdot_ref.assign([alphaX * axis_x, alphaY * axis_y, 0], nodes=ns_srbd)
        #w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]], nodes=ns)
        #orientation_tracking_gain.assign(cs.sqrt(1e5) if rotate else 0.)
    else:
        axis_x = keyboard.is_pressed('up') - keyboard.is_pressed('down')
        axis_y = keyboard.is_pressed('right') - keyboard.is_pressed('left')

        lip.rdot_ref.assign([alphaX * axis_x, alphaY * axis_y, 0], nodes=ns_lip)
        #w_ref.assign([0, 0, 0], nodes=ns_srbd)
        #orientation_tracking_gain.assign(0.)

    if motion == "walking":
        wpg.set("step")
    elif motion == "jumping":
        wpg.set("jump")
    else:
        wpg.set("standing")

    # solve
    tic()
    meta_solver.solve()
    solution_time_pub.publish(toc())
    solution = meta_solver.getSolutionDict()

    lip_solution = meta_solver.getSolutionModel(1)

    t = rospy.Time().now()

    lip_input = lip_solution["u_opt"][:, 0]
    lip_state = lip_solution["x_opt"][:, 0]
    utilities.ZMPTfBroadcaster(lip_solution['z'][:, 0], t)
    #print("zmp: ", lip_solution['z'][2, :])
    #print("com: ", lip_solution['r'][2, :])
    #exit()

    rddot0 = lip.RDDOT(lip_state, lip_input, solver_lip.get_params_value(0))
    fzmp = lip.m * (np.array([0., 0., 9.81]) + rddot0)
    viz.publishContactForce(t, fzmp, 'ZMP')
   # for i in range(0, lip.nc):
   #     viz.publishPointTrj(lip_solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
   # viz.SRBDViewer(srbd.I, "SRB", t, lip.nc)  # TODO: should we use w_R_b * I * w_R_b.T?
    viz.publishPointTrj(lip_solution["r"], t, name="COM", frame="world", color=[1., 1., 0.], namespace="LIP")
    viz.publishPointTrj(lip_solution["z"], t, name="ZMP", frame="world", color=[0., 1., 1.], namespace="LIP")


    c0_hist = dict()
    for i in range(0, srbd.nc):
        c0_hist['c' + str(i)] = solution['c' + str(i)][:, 0]

    utilities.SRBDTfBroadcaster(solution['r'][:, 0], solution['o'][:, 0], c0_hist, t)
    for i in range(0, srbd.nc):
        viz.publishContactForce(t, srbd.force_scaling * solution['f' + str(i)][:, 0], 'c' + str(i))
        viz.publishPointTrj(solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
    viz.SRBDViewer(srbd.I, "SRB", t, srbd.nc)  # TODO: should we use w_R_b * I * w_R_b.T?
    viz.publishPointTrj(solution["r"], t, "SRB", "world")

    utilities.visualize_horizon([5, 10], solution, srbd.nc, t, srbd.I)
    Inertia = srbd.I[0, 0] * np.identity(3)
    utilities.visualize_horizon([0, 5, 10], lip_solution, srbd.nc, t, Inertia, body_name="LIP", off_set=1000)

    cc = dict()
    ff = dict()
    for i in range(0, srbd.nc):
        cc[i] = solution["c" + str(i)][:, 0]
        ff[i] = solution["f" + str(i)][:, 0]

    # simulation integration
    input = solution["u_opt"][:, 0]
    srbd_state = np.array(cs.DM(srbd_euler_integrator(srbd_state, input, solver_srbd.get_params_value(0))))
    srbd_state[3:7] /= cs.norm_2(srbd_state[3:7])
    # print(f"state:", solution["x_opt"])
    # print(f"input:", solution["u_opt"])
    rddot0 = srbd.RDDOT(input)
    wdot0 = srbd.WDOT(srbd_state, input)

    w_R_b0 = utils.toRot(srbd_state[3:7])
    srbd_0 = kin_dyn.SRBD(srbd.m / srbd.force_scaling, w_R_b0 * srbd.I / srbd.force_scaling * w_R_b0.T, ff,
                          solution["r"][:, 0], rddot0, cc, solution["w"][:, 0], wdot0)
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