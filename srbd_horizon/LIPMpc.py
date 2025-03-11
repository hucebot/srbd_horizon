from srbd_horizon.mpc import *
import numpy as np
import casadi as cs
import scipy
from ttictoc import tic, toc

class LipController(MpcController):
    def __init__(self, initial_joint_state, ns, T, opts=dict()):
        MpcController.__init__(self, initial_joint_state)

        self.max_iteration = rospy.get_param("max_iteration", 20)
        print(f"max_iteration: {self.max_iteration}")

        self.ns = ns

        self.lip = LIPProblem.LIPProblem()
        self.lip.createLIPProblem(ns, T, initial_joint_state)

        if opts == dict():
            opts["max_iters"] = 100
            opts["alpha_converge_threshold"] = 1e-12
            opts["beta"] = 1e-3
        self.solver = ddp.DDPSolver(self.lip.prb, opts=opts)

        self.state = self.lip.getInitialState()
        self.x_warmstart = np.zeros((self.state.shape[0], ns + 1))
        for i in range(0, ns + 1):
            self.x_warmstart[:, i] = self.state
        self.u_warmstart = np.zeros((self.lip.getStaticInput().shape[0], ns))
        for i in range(0, ns):
            self.u_warmstart[:, i] = self.lip.getStaticInput()

        np.set_printoptions(suppress=True)

        dae = dict()
        dae["x"] = cs.vertcat(self.lip.prb.getState().getVars())
        dae["ode"] = self.lip.prb.getDynamics()
        dae["p"] = cs.vertcat(self.lip.prb.getInput().getVars())
        dae["quad"] = 0.
        self.simulation_euler_integrator = self.solver.get_f(0)

        self.wpg = wpg.steps_phase(number_of_legs=2, contact_model=self.lip.contact_model,
                              c_init_z=self.lip.initial_foot_position[0][2].__float__())

        self.solution_time_vec = list()

    def __del__(self):
        scipy.io.savemat('dlip_solution_time.mat', {'solution_time': np.array(self.solution_time_vec)})

    def solve(self, state=None):
        if state is not None:
            self.state = state

        self.solver.setInitialState(self.state)

        # shift reference velocities back by one node
        self.lip.shiftReferences()

        self.lip.assignReferences(self.alphaX * self.axis_x, self.alphaY * self.axis_y, 0)

        self.lip.shiftContactConstraints()
        self.lip.setAction(self.motion, self.wpg)

        # solve
        tic()
        self.solver.solve()
        solution_time = toc()
        self.solution_time_vec.append(solution_time)
        self.solution_time_pub.publish(solution_time)
        self.solution = self.solver.getSolutionDict()


        input = self.solution["u_opt"][:, 0]
        self.state = np.array(cs.DM(self.simulation_euler_integrator(self.state, input, self.solver.get_params_value(0))))

        self.rddot0 = self.lip.RDDOT(self.state, input, self.solver.get_params_value(0))
        self.fzmp = self.lip.m * (np.array([0., 0., 9.81]) + self.rddot0)

        cc = dict()
        for i in range(0, self.lip.nc):
            cc[i] = self.solution["c" + str(i)][:, 0]

        self.ret["state"] = state
        self.ret["input"] = input
        self.ret["rddot0"] = self.rddot0
        self.ret["fzmp"] = self.fzmp
        self.ret["cc"] = cc

    def visualize(self):
        #c0_hist = dict()
        #for i in range(0, self.lip.nc):
        #    c0_hist['c' + str(i)] = self.solution['c' + str(i)][:, 0]

        t = rospy.Time().now()

        nodes_to_visualize = np.arange(0, self.lip.prb.getNNodes(), 6).tolist()
        I = list()
        for n in nodes_to_visualize:
            I.append(0.1 * np.eye(3))
        viz.visualize_horizon(nodes_to_visualize, self.solution, self.lip.nc, t, Inertia=I, body_name="SRB", offset=100, scale=0.5)


        #utilities.SRBDTfBroadcaster(self.solution['r'][:, 0], np.array([0., 0., 0., 1.]), c0_hist, t)
        utilities.ZMPTfBroadcaster(self.solution['z'][:, 0], t)

        viz.publishContactForce(t, self.fzmp, 'ZMP')
        for i in range(0, self.lip.nc):
            viz.publishPointTrj(self.solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
        #viz.SRBDViewer(np.eye(3), "SRB", t, self.lip.nc)
        viz.publishPointTrj(self.solution["r"], t, "SRB", "world")
        viz.publishPointTrj(self.solution["z"], t, name="ZMP", frame="world", color=[0., 1., 1.], namespace="LIP")


