from srbd_horizon.mpc import *
import numpy as np
import casadi as cs
import scipy
from ttictoc import tic, toc

class SRBDController(MpcController):
    def __init__(self, initial_joint_state, ns, T, opts=dict()):
        MpcController.__init__(self, initial_joint_state)

        self.ns = ns

        max_iteration = rospy.get_param("max_iteration", 20)
        print(f"max_iteration: {max_iteration}")

        self.solution_time_vec = list()

        self.srbd = SRBDProblem.SRBDProblem()
        self.srbd.createSRBDProblem(ns, T)

        self.solution_time_pub = rospy.Publisher("solution_time", Float32, queue_size=10)
        self.srbd_pub = rospy.Publisher("srbd_constraint", WrenchStamped, queue_size=10)
        self.srbd_msg = WrenchStamped()

        if opts == dict():
            opts["max_iters"] = 100
            opts["alpha_converge_threshold"] = 1e-12
            opts["beta"] = 1e-3
        self.solver = ddp.DDPSolver(self.srbd.prb, opts=opts)

        # set initial state and warmstart ddp
        self.state = self.srbd.getInitialState()
        self.x_warmstart = np.zeros((self.state.shape[0], ns + 1))
        for i in range(0, ns + 1):
            self.x_warmstart[:, i] = self.state
        self.u_warmstart = np.zeros((self.srbd.getStaticInput().shape[0], ns))
        for i in range(0, ns):
            self.u_warmstart[:, i] = self.srbd.getStaticInput()

        # define discrete dynamics
        dae = dict()
        dae["x"] = cs.vertcat(self.srbd.prb.getState().getVars())
        dae["ode"] = self.srbd.prb.getDynamics()
        dae["p"] = cs.vertcat(self.srbd.prb.getInput().getVars())
        dae["quad"] = 0.
        self.simulation_euler_integrator = self.solver.get_f(0)

        # Walking patter generator and scheduler
        self.wpg = wpg.steps_phase(number_of_legs=2, contact_model=self.srbd.contact_model,
                              c_init_z=self.srbd.initial_foot_position[0][2].__float__())

    def __del__(self):
        scipy.io.savemat('dsrbd_solution_time.mat', {'solution_time': np.array(self.solution_time_vec)})

    def solve(self, state=None):
        if state is not None:
            self.state = state

        self.solver.setInitialState(self.state)

        # shift reference velocities back by one node
        self.srbd.shiftReferences()

        if self.wx is None:
            self.srbd.assignReferences(self.alphaX * self.axis_x, self.alphaY * self.axis_y, 0.)
        else:
            self.srbd.assignVWReferences(self.alphaX * self.axis_x, self.alphaY * self.axis_y, 0, self.wx, self.wy, self.wz)

        self.srbd.shiftContactConstraints()
        self.srbd.setAction(self.motion, self.wpg)

        # solve
        tic()
        self.solver.solve()
        solution_time = toc()
        self.solution_time_pub.publish(solution_time)
        self.solution_time_vec.append(solution_time)
        self.solution = self.solver.getSolutionDict()

        self.c0_hist = dict()
        for i in range(0, self.srbd.nc):
            self.c0_hist['c' + str(i)] = self.solution['c' + str(i)][:, 0]


        cc = dict()
        ff = dict()
        for i in range(0, self.srbd.nc):
            cc[i] = self.solution["c" + str(i)][:, 0]
            ff[i] = self.solution["f" + str(i)][:, 0]

        # simulation integration
        input = self.solution["u_opt"][:, 0]
        self.state = np.array(
            cs.DM(self.simulation_euler_integrator(self.state, input, cs.vcat(list(self.srbd.prb.getParameters().values())))))
        self.state[3:7] /= cs.norm_2(self.state[3:7])
        # print(f"state:", solution["x_opt"])
        # print(f"input:", solution["u_opt"])
        rddot0 = self.srbd.RDDOT(input)
        wdot0 = self.srbd.WDOT(self.state, input)

        self.w_R_b0 = utils.toRot(self.state[3:7])
        self.Iw0 = np.matmul(np.matmul(self.w_R_b0, self.srbd.I / self.srbd.force_scaling), self.w_R_b0.T)
        self.srbd_0 = kin_dyn.SRBD(self.srbd.m / self.srbd.force_scaling, self.Iw0, ff,
                              self.solution["r"][:, 0], rddot0, cc, self.solution["w"][:, 0], wdot0)

        self.ret["state"] = self.state
        self.ret["input"] = input
        self.ret["rddot0"] = rddot0
        self.ret["wdot0"] = wdot0
        self.ret["cc"] = cc
        self.ret["ff"] = ff

    def visualize(self):
        t = rospy.Time().now()
        utilities.SRBDTfBroadcaster(self.solution['r'][:, 0], self.solution['o'][:, 0], self.c0_hist, t)
        for i in range(0, self.srbd.nc):
            viz.publishContactForce(t, self.srbd.force_scaling * self.solution['f' + str(i)][:, 0], 'c' + str(i))
            viz.publishPointTrj(self.solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
        viz.SRBDViewer(self.Iw0, "SRB", t, self.srbd.nc)
        viz.publishPointTrj(self.solution["r"], t, "SRB", "world")

        self.srbd_msg.header.stamp = t
        self.srbd_msg.wrench.force.x = self.srbd_0[0]
        self.srbd_msg.wrench.force.y = self.srbd_0[1]
        self.srbd_msg.wrench.force.z = self.srbd_0[2]
        self.srbd_msg.wrench.torque.x = self.srbd_0[3]
        self.srbd_msg.wrench.torque.y = self.srbd_0[4]
        self.srbd_msg.wrench.torque.z = self.srbd_0[5]
        self.srbd_pub.publish(self.srbd_msg)




