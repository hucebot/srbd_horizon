from srbd_horizon.mpc import *
import numpy as np
import casadi as cs
import scipy
from ttictoc import tic, toc

class ModelSchedulingController(MpcController):
    def __init__(self, initial_joint_state, ns, T, opts=dict()):
        MpcController.__init__(self, initial_joint_state)

        self.ns_srbd = ns['SRBD']
        ns_lip = ns['LIP']
        T_srbd = T['SRBD']
        T_lip = T['LIP']

        self.srbd = SRBDProblem.SRBDProblem()
        self.srbd.createSRBDProblem(self.ns_srbd, T_srbd, initial_joint_state)
        self.lip = LIPProblem.LIPProblem()
        self.lip.createLIPProblem(ns_lip, T_lip, initial_joint_state)

        self.srbd_pub = rospy.Publisher("srbd_constraint", WrenchStamped, queue_size=10)
        self.srbd_msg = WrenchStamped()

        self.solution_time_vec = list()

        opts = dict()
        opts["max_iters"] = 100
        opts["alpha_converge_threshold"] = 1e-12
        opts["beta"] = 1e-3
        self.solver_srbd = ddp.DDPSolver(self.srbd.prb, opts=opts)
        self.solver_lip = ddp.DDPSolver(self.lip.prb, opts=opts)

        # set initial state and warmstart ddp
        lip_state = self.lip.getInitialState()
        lip_x_warmstart = np.zeros((lip_state.shape[0], ns_lip + 1))
        for i in range(0, ns_lip + 1):
            lip_x_warmstart[:, i] = lip_state
        lip_u_warmstart = np.zeros((self.lip.getStaticInput().shape[0], ns_lip))
        for i in range(0, ns_lip):
            lip_u_warmstart[:, i] = self.lip.getStaticInput()

        # set initial state and warmstart ddp
        self.srbd_state = self.srbd.getInitialState()
        srbd_x_warmstart = np.zeros((self.srbd_state.shape[0], self.ns_srbd + 1))
        for i in range(0, self.ns_srbd + 1):
            srbd_x_warmstart[:, i] = self.srbd_state
        srbd_u_warmstart = np.zeros((self.srbd.getStaticInput().shape[0], self.ns_srbd))
        for i in range(0, self.ns_srbd):
            srbd_u_warmstart[:, i] = self.srbd.getStaticInput()

        self.solver_srbd.set_x_warmstart(srbd_x_warmstart)
        self.solver_srbd.set_u_warmstart(srbd_u_warmstart)
        self.solver_lip.set_x_warmstart(lip_x_warmstart)
        self.solver_lip.set_u_warmstart(lip_u_warmstart)

        # define discrete dynamics
        dae = dict()
        dae["x"] = cs.vertcat(self.srbd.prb.getState().getVars())
        dae["ode"] = self.srbd.prb.getDynamics()
        dae["p"] = cs.vertcat(self.srbd.prb.getInput().getVars())
        dae["quad"] = 0.
        self.srbd_euler_integrator = self.solver_srbd.get_f(0)

        self.meta_solver = ddp.MetaSolver(self.srbd.prb, None)

        model_mapping_function = cs.Function("srbd_to_lip", [self.srbd.prb.getState().getVars()], [cs.vcat(
            [self.srbd.prb.getState().getVars()[0:3], self.srbd.prb.getState().getVars()[7:19],
             self.srbd.prb.getState().getVars()[19:22], self.srbd.prb.getState().getVars()[25:37]])])
        self.meta_solver.add(self.solver_srbd, model_mapping_function)
        foo_mapping_function = cs.Function("foo", [self.lip.prb.getState().getVars()], [cs.DM.zeros(1, 1)])
        self.meta_solver.add(self.solver_lip, foo_mapping_function)
        self.meta_solver.setMaxIterations(1)

        self.lip_wpg = wpg.steps_phase(number_of_legs=2, contact_model=self.lip.contact_model,
                                  c_init_z=self.lip.initial_foot_position[0][2].__float__())

    def __del__(self):
        scipy.io.savemat('model_scheduling_solution_time.mat', {'solution_time': np.array(self.solution_time_vec)})

    def solve(self, state=None):
        if state is not None:
            self.srbd_state = state

        self.meta_solver.setInitialState(self.srbd_state)

        self.srbd.shiftReferences(self.ns_srbd)
        self.srbd.shiftContactConstraints(end_node=self.ns_srbd)


        self.srbd.rdot_ref.assign(self.lip.rdot_ref.getValues(nodes=0), nodes=self.ns_srbd - 1)
        for i in range(0, self.srbd.nc):
            self.srbd.cdot_switch[i].assign(self.lip.cdot_switch[i].getValues(nodes=0), nodes=self.ns_srbd - 1)
            self.srbd.c_ref[i].assign(self.lip.c_ref[i].getValues(nodes=0), nodes=self.ns_srbd - 1)
        # w_ref??
        # oref??

        self.lip.shiftReferences()
        self.lip.shiftContactConstraints()
        self.lip.assignReferences(self.alphaX * self.axis_x, self.alphaY * self.axis_y, 0)

        self.lip.setAction(self.motion, self.lip_wpg)

        # solve
        tic()
        self.meta_solver.solve()
        solution_time = toc()
        self.solution_time_pub.publish(solution_time)
        self.solution_time_vec.append(solution_time)
        self.solution = self.meta_solver.getSolutionDict()

        self.lip_solution = self.meta_solver.getSolutionModel(1)

        t = rospy.Time().now()

        lip_input = self.lip_solution["u_opt"][:, 0]
        lip_state = self.lip_solution["x_opt"][:, 0]
        utilities.ZMPTfBroadcaster(self.lip_solution['z'][:, 0], t)
        # print("zmp: ", lip_solution['z'][2, :])
        # print("com: ", lip_solution['r'][2, :])
        # exit()

        rddot0 = self.lip.RDDOT(lip_state, lip_input, self.solver_lip.get_params_value(0))
        fzmp = self.lip.m * (np.array([0., 0., 9.81]) + rddot0)
        viz.publishContactForce(t, fzmp, 'ZMP')
        # for i in range(0, lip.nc):
        #     viz.publishPointTrj(lip_solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
        # viz.SRBDViewer(srbd.I, "SRB", t, lip.nc)  # TODO: should we use w_R_b * I * w_R_b.T?
        viz.publishPointTrj(self.lip_solution["r"], t, name="COM", frame="world", color=[1., 1., 0.], namespace="LIP")
        viz.publishPointTrj(self.lip_solution["z"], t, name="ZMP", frame="world", color=[0., 1., 1.], namespace="LIP")

        c0_hist = dict()
        for i in range(0, self.srbd.nc):
            c0_hist['c' + str(i)] = self.solution['c' + str(i)][:, 0]

        utilities.SRBDTfBroadcaster(self.solution['r'][:, 0], self.solution['o'][:, 0], c0_hist, t)
        for i in range(0, self.srbd.nc):
            viz.publishContactForce(t, self.srbd.force_scaling * self.solution['f' + str(i)][:, 0], 'c' + str(i))
            viz.publishPointTrj(self.solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
        viz.SRBDViewer(self.srbd.I, "SRB", t, self.srbd.nc)  # TODO: should we use w_R_b * I * w_R_b.T?
        viz.publishPointTrj(self.solution["r"], t, "SRB", "world")

        cc = dict()
        ff = dict()
        for i in range(0, self.srbd.nc):
            cc[i] = self.solution["c" + str(i)][:, 0]
            ff[i] = self.solution["f" + str(i)][:, 0]

        # simulation integration
        input = self.solution["u_opt"][:, 0]
        self.srbd_state = np.array(cs.DM(self.srbd_euler_integrator(self.srbd_state, input, self.solver_srbd.get_params_value(0))))
        self.srbd_state[3:7] /= cs.norm_2(self.srbd_state[3:7])
        # print(f"state:", solution["x_opt"])
        # print(f"input:", solution["u_opt"])
        rddot0 = self.srbd.RDDOT(input)
        wdot0 = self.srbd.WDOT(self.srbd_state, input)

        w_R_b0 = utils.toRot(self.srbd_state[3:7])
        srbd_0 = kin_dyn.SRBD(self.srbd.m / self.srbd.force_scaling, w_R_b0 * self.srbd.I / self.srbd.force_scaling * w_R_b0.T, ff,
                              self.solution["r"][:, 0], rddot0, cc, self.solution["w"][:, 0], wdot0)
        self.srbd_msg.header.stamp = t
        self.srbd_msg.wrench.force.x = srbd_0[0]
        self.srbd_msg.wrench.force.y = srbd_0[1]
        self.srbd_msg.wrench.force.z = srbd_0[2]
        self.srbd_msg.wrench.torque.x = srbd_0[3]
        self.srbd_msg.wrench.torque.y = srbd_0[4]
        self.srbd_msg.wrench.torque.z = srbd_0[5]
        self.srbd_pub.publish(self.srbd_msg)

    def visualize(self):
        pass

