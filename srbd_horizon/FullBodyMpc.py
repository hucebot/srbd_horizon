from srbd_horizon.mpc import *
import numpy as np
import casadi as cs
import scipy
from ttictoc import tic, toc


class fullModelController(MpcController):
    def __init__(self, initial_joint_state, ns, T, opts=dict()):
        MpcController.__init__(self, initial_joint_state)
        self.ns = ns
        self.max_iteration = rospy.get_param("max_iteration", 1)
        print(f"max_iteration: {self.max_iteration}")

        self.full_model = FullBodyProblem.FullBodyProblem()
        self.full_model.createFullBodyProblem(ns, T, initial_joint_state, include_transmission_forces=False)

        self.joint_state_publisher = rospy.Publisher("joint_states", JointState, queue_size=10)

        self.solution_time_vec = list()

        nx = [self.full_model.nx] * (ns + 1)
        nu = [self.full_model.nu] * ns
        nu.append(0)
        ng = self.full_model.getNonDynamicConstraintList()
        print(f"nx: {nx}")
        print(f"nu: {nu}")
        print(f"ng: {ng}")

        solver = 'osqp'
        if opts == dict():
            if solver == 'osqp':
                opts = {"gnsqp.max_iter": self.max_iteration,
                        'gnsqp.osqp.scaled_termination': False,
                        # 'gnsqp.osqp.check_termination': 0,
                        # 'gnsqp.osqp.alpha': 1.9,
                        # 'gnsqp.osqp.rho': 1.,
                        # 'gnsqp.osqp.scaling': 10,
                        'gnsqp.eps_regularization': 1e-5,  # 1e-2,
                        'gnsqp.osqp.polish': False,
                        'gnsqp.jit': True,
                        'gnsqp.osqp.linsys_solver_mkl_pardiso': False,
                        'gnsqp.osqp.verbose': False,
                        # "gnsqp.osqp.adaptive_rho": False,
                        # "gnsqp.osqp.rho": 1e-2,
                        # "gnsqp.osqp.eps_abs": 1e-3,
                        # "gnsqp.osqp.eps_rel": 1e-3,
                        # "gnsqp.osqp.eps_prim_inf": 1e-3,
                        # "gnsqp.osqp.eps_dual_inf": 1e-3,
                        }
            elif solver == 'fatrop':
                fatrop_opts = {"warm_start_init_point": True,
                               "iterative_refinement": False,
                               "mu_init": 1e-5,
                               # "max_iter": 20,
                               "accept_every_trial_step": True,
                               "tol": 1e-3}
                opts = {"gnsqp.structure_detection": "auto",
                        # "gnsqp.N": ns, "gnsqp.nx": nx, "gnsqp.nu": nu, "gnsqp.ng": ng,
                        'gnsqp.eps_regularization': 1e-6,
                        'gnsqp.error_on_fail': False,
                        'gnsqp.debug': False,
                        "gnsqp.max_iter": self.max_iteration,
                        "gnsqp.fatrop": fatrop_opts
                        }
            elif solver == 'hpipm':
                hpipm_opts = {"warm_start": True,
                              "mode": "speed",
                              "mu0": 1e-6,
                              "abs_form": True,
                              "comp_dual_sol_eq": 1e-3,
                              "iter_max": 200,
                              }
                opts = {"gnsqp.max_iter": self.max_iteration,
                        'gnsqp.eps_regularization': 1e-6,
                        "gnsqp.N": ns, "gnsqp.nx": nx, "gnsqp.nu": nu, "gnsqp.ng": ng,
                        "gnsqp.verbose": False,
                        'gnsqp.jit': True,
                        "gnsqp.error_on_fail": False,
                        # 'gnsqp.hpipm': hpipm_opts
                        }
            elif solver == 'proxqp':
                proxqp_opts = {"verbose": True,
                               "eps_abs": 1e-4, "eps_rel": 1e-4,
                               "backend": "sparse"}
                opts = {"gnsqp.max_iter": self.max_iteration,
                        'gnsqp.eps_regularization': 1e-6,
                        'gnsqp.jit': True,
                        "gnsqp.error_on_fail": False,
                        "gnsqp.warm_start_primal": True,
                        "gnsqp.warm_start_dual": True,
                        'gnsqp.proxqp': proxqp_opts
                        }

        self.solver = ddp.SQPSolver(self.full_model.prb, qp_solver_plugin=solver, opts=opts)
        self.full_model.q.setInitialGuess(self.full_model.getInitialState()[0:self.full_model.nq])
        self.full_model.qdot.setInitialGuess(self.full_model.getInitialState()[self.full_model.nq:])
        self.full_model.qddot.setInitialGuess(self.full_model.getStaticInput()[0:self.full_model.nv])
        i = -1
        for foot_frame in self.full_model.foot_frames:
            i += 1
            self.full_model.f[foot_frame].setInitialGuess(
                self.full_model.getStaticInput()[self.full_model.nv + i * 3:self.full_model.nv + i * 3 + 3])
        if self.full_model.include_transmission_forces:
            self.full_model.left_actuation_lambda.setInitialGuess(
                self.full_model.getStaticInput()[self.full_model.nv + i * 3 + 3:self.full_model.nv + i * 3 + 3 + 2])
            self.full_model.right_actuation_lambda.setInitialGuess(
                self.full_model.getStaticInput()[self.full_model.nv + i * 3 + 3 + 2:])

        self.solver.setInitialGuess(self.full_model.getInitialGuess())

        """
        Dictionary to store variables used for warm-start
        """
        self.variables_dict = {"q": self.full_model.q, "qdot": self.full_model.qdot, "qddot": self.full_model.qddot}
        if self.full_model.include_transmission_forces:
            self.variables_dict["left_actuation_lambda"] = self.full_model.left_actuation_lambda
            self.variables_dict["right_actuation_lambda"] = self.full_model.right_actuation_lambda
        for foot_frame in self.full_model.foot_frames:
            self.variables_dict["f_" + foot_frame] = self.full_model.f[foot_frame]

        self.solver.solve()
        self.solution = self.solver.getSolutionDict()
        self.solution['q'] = utilities.normalize_quaternion_part_horizon(self.solution['q'], ns)

        self.joint_state_msg = JointState()
        self.joint_state_msg.name = self.full_model.kindyn.joint_names()[2:]

        k = 0
        initial_foot_position = dict()
        # cdot_switch = dict()
        for foot_frame in self.full_model.foot_soles:
            initial_foot_position[k] = self.full_model.initial_foot_position[foot_frame]
            # cdot_switch[k] = full_model.cdot_switch[foot_frame]
            k += 1

        self.wpg = wpg.steps_phase(number_of_legs=2, contact_model=self.full_model.contact_model,
                                   c_init_z=initial_foot_position[0][2].__float__())

    def __del__(self):
        scipy.io.savemat('full_body_solution_time.mat', {'solution_time': np.array(self.solution_time_vec)})

    def solve(self, state=None):
        if state is not None:
            self.state = state

        """
            Automatically set initial guess from solution to variables in variables_dict
            """
        mat_storer.setInitialGuess(self.variables_dict, self.solution)
        self.solver.setInitialGuess(self.full_model.getInitialGuess())
        # open loop
        self.full_model.q.setBounds(self.solution['q'][:, 1], self.solution['q'][:, 1], 0)
        self.full_model.qdot.setBounds(self.solution['qdot'][:, 1], self.solution['qdot'][:, 1], 0)

        self.full_model.shiftReferences()

        if self.wx is None:
            self.full_model.assignReferences(self.alphaX * self.axis_x, self.alphaY * self.axis_y, 0.)
        else:
            self.full_model.assignVWReferences(self.alphaX * self.axis_x, self.alphaY * self.axis_y, 0, self.wx, self.wy,
                                         self.wz)

        self.full_model.shiftContactConstraints()
        self.full_model.setAction(self.motion, self.wpg)

        # solve
        tic()
        self.solver.solve()
        solution_time = toc()
        self.solution_time_pub.publish(solution_time)
        self.solution_time_vec.append(solution_time)
        self.solution = self.solver.getSolutionDict()
        self.solution['q'] = utilities.normalize_quaternion_part_horizon(self.solution['q'], self.ns)



        # publish contact forces and contact points
        self.c = dict()
        for foot_frame in self.full_model.foot_frames:
            C = self.full_model.kindyn.fk(foot_frame)
            self.c[foot_frame] = np.zeros((3, self.ns + 1))
            for i in range(0, self.ns + 1):
                self.c[foot_frame][:, i] = C(q=self.solution['q'][:, i])['ee_pos'].toarray().flatten()

        # publish center of mass
        COM = self.full_model.kindyn.centerOfMass()
        self.com = np.zeros((3, self.ns + 1))
        for i in range(0, self.ns + 1):
            self.com[:, i] = COM(q=self.solution['q'][:, i])['com'].toarray().flatten()

        #self.ret["state"] = self.state
        self.ret["solution"] = self.solution
        self.ret["c"] = self.c
        self.ret["com"] = self.com


    def visualize(self):
        t = rospy.Time.now()
        # publish tf
        br = tf.TransformBroadcaster()
        br.sendTransform(self.solution['q'][0:3, 1], self.solution['q'][3:7, 1], t, "base_link", "world")

        # publish joint states
        self.joint_state_msg.position = self.solution['q'][7:, 1]
        self.joint_state_msg.header.stamp = t
        self.joint_state_publisher.publish(self.joint_state_msg)

        for i in range(0, self.full_model.nc):
            viz.publishContactForce(t, self.full_model.force_scaling * self.solution[
                                                                           'f_' + self.full_model.foot_frames[i]][:, 0],
                                    frame=self.full_model.foot_frames[i], topic='fc' + str(i))
            viz.publishPointTrj(self.c[self.full_model.foot_frames[i]], t, 'c' + str(i), "world", color=[0., 0., 1.])

        viz.publishPointTrj(self.com, t, "SRB", "world")





