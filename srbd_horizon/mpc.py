import scipy
from ttictoc import tic,toc
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float32
from srbd_horizon import viz, wpg, ddp, utilities, prb
import numpy as np
import keyboard
import rospy
import casadi as cs
from horizon.utils import mat_storer
import tf


def joy_cb(msg):
    global joy_msg
    joy_msg = msg

class MpcController:
    def __init__(self, initial_joint_state):
        self.initial_joint_state = initial_joint_state

        rospy.init_node('mpc_controller', anonymous=True)

        self.solution_time_pub = rospy.Publisher("solution_time", Float32, queue_size=10)
        # game controller
        rospy.Subscriber('/joy', Joy, joy_cb)
        global joy_msg
        joy_msg = None

    def get_solution(self, state=None):
        raise NotImplementedError()

class fullModelController(MpcController):
    def __init__(self, initial_joint_state, ns, T, opts=dict()):
        MpcController.__init__(self, initial_joint_state)
        self.ns = ns
        self.max_iteration = rospy.get_param("max_iteration", 1)
        print(f"max_iteration: {self.max_iteration}")

        self.full_model = prb.FullBodyProblem()
        self.full_model.createFullBodyProblem(ns, T, include_transmission_forces=False)

        self.joint_state_publisher = rospy.Publisher("joint_states", JointState, queue_size=10)

        solver = 'osqp'
        if opts == dict():
            if solver == 'osqp':
                opts = {"gnsqp.max_iter": self.max_iteration,
                        'gnsqp.osqp.scaled_termination': False,
                        'gnsqp.eps_regularization': 1e-6,  # 1e-2,
                        'gnsqp.osqp.polish': False,
                        'gnsqp.osqp.linsys_solver_mkl_pardiso': True,
                        'gnsqp.osqp.verbose': False}
            elif solver == 'fatrop':
                nx = [self.full_model.nx] * (ns + 1)
                nu = [self.full_model.nu] * ns
                nu.append(0)
                ng = self.full_model.getNonDynamicConstraintList()
                print(f"nx: {nx}")
                print(f"nu: {nu}")
                print(f"ng: {ng}")
                opts = {"gnsqp.structure_detection": "manual",
                        "gnsqp.N": ns,
                        "gnsqp.nx": nx,
                        "gnsqp.nu": nu,
                        "gnsqp.ng": ng,
                        'gnsqp.eps_regularization': 1e-6,
                        "gnsqp.max_iter": self.max_iteration}

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
        None

    def get_solution(self, state=None):
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

        # references
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
        for j in range(1, self.ns + 1):
            self.full_model.rdot_ref.assign(self.full_model.rdot_ref.getValues(nodes=j), nodes=j - 1)
            self.full_model.w_ref.assign(self.full_model.w_ref.getValues(nodes=j), nodes=j - 1)
            self.full_model.oref.assign(self.full_model.oref.getValues(nodes=j), nodes=j - 1)
            self.full_model.orientation_tracking_gain.assign(self.full_model.orientation_tracking_gain.getValues(nodes=j),
                                                        nodes=j - 1)

        # assign new references based on user input
        if motion == "standing":
            alphaX, alphaY = 0.1, 0.1
        else:
            alphaX, alphaY = 0.5, 0.5

        if joy_msg is not None:
            self.full_model.rdot_ref.assign([alphaX * joy_msg.axes[1], alphaY * joy_msg.axes[0], 0.1 * joy_msg.axes[7]],
                                       nodes=self.ns)
            # w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]], nodes=ns)
            # orientation_tracking_gain.assign(cs.sqrt(1e5) if rotate else 0.)
        else:
            axis_x = keyboard.is_pressed('up') - keyboard.is_pressed('down')
            axis_y = keyboard.is_pressed('right') - keyboard.is_pressed('left')

            self.full_model.rdot_ref.assign([alphaX * axis_x, alphaY * axis_y, 0], nodes=self.ns)
            # w_ref.assign([0, 0, 0], nodes=ns)
            # orientation_tracking_gain.assign(0.)

        self.full_model.shiftContactConstraints()
        self.full_model.setAction(motion, self.wpg)

        # solve
        tic()
        self.solver.solve()
        self.solution_time_pub.publish(toc())
        self.solution = self.solver.getSolutionDict()
        self.solution['q'] = utilities.normalize_quaternion_part_horizon(self.solution['q'], self.ns)

        t = rospy.Time.now()
        # publish tf
        br = tf.TransformBroadcaster()
        br.sendTransform(self.solution['q'][0:3, 1], self.solution['q'][3:7, 1], t, "base_link", "world")

        # publish joint states
        self.joint_state_msg.position = self.solution['q'][7:, 1]
        self.joint_state_msg.header.stamp = t
        self.joint_state_publisher.publish(self.joint_state_msg)

        # publish contact forces and contact points
        c = dict()
        for foot_frame in self.full_model.foot_frames:
            C = self.full_model.kindyn.fk(foot_frame)
            c[foot_frame] = np.zeros((3, self.ns + 1))
            for i in range(0, self.ns + 1):
                c[foot_frame][:, i] = C(q=self.solution['q'][:, i])['ee_pos'].toarray().flatten()
        for i in range(0, self.full_model.nc):
            viz.publishContactForce(t, self.solution['f_' + self.full_model.foot_frames[i]][:, 0],
                                    frame=self.full_model.foot_frames[i], topic='fc' + str(i))
            viz.publishPointTrj(c[self.full_model.foot_frames[i]], t, 'c' + str(i), "world", color=[0., 0., 1.])

        # publish center of mass
        COM = self.full_model.kindyn.centerOfMass()
        com = np.zeros((3, self.ns + 1))
        for i in range(0, self.ns + 1):
            com[:, i] = COM(q=self.solution['q'][:, i])['com'].toarray().flatten()
        viz.publishPointTrj(com, t, "SRB", "world")

        return self.solution


class LipController(MpcController):
    def __init__(self, initial_joint_state, ns, T, opts=dict()):
        MpcController.__init__(self, initial_joint_state)

        self.max_iteration = rospy.get_param("max_iteration", 20)
        print(f"max_iteration: {self.max_iteration}")

        self.ns = ns

        self.lip = prb.LIPProblem()
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


    def get_solution(self, state=None):
        if state is not None:
            self.state = state

        self.solver.setInitialState(self.state)

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
        for j in range(1, self.ns + 1):
            self.lip.rdot_ref.assign(self.lip.rdot_ref.getValues(nodes=j), nodes=j - 1)
            self.lip.eta2_p.assign(self.lip.eta2_p.getValues(nodes=j), nodes=j - 1)

        if self.lip.cdot_switch[0].getValues(self.ns) == 0 and self.lip.cdot_switch[1].getValues(self.ns) == 0 and self.lip.cdot_switch[
            2].getValues(self.ns) == 0 and self.lip.cdot_switch[3].getValues(self.ns) == 0:
            self.lip.eta2_p.assign(0., nodes=self.ns)
        else:
            self.lip.eta2_p.assign(self.lip.eta2, nodes=self.ns)

        # assign new references based on user input
        if motion == "standing":
            alphaX, alphaY = 0.1, 0.1
        else:
            alphaX, alphaY = 0.5, 0.5

        if joy_msg is not None:
            self.lip.rdot_ref.assign([alphaX * joy_msg.axes[1], alphaY * joy_msg.axes[0], 0.1 * joy_msg.axes[7]], nodes=self.ns)
            # w_ref.assign([1. * joy_msg.axes[6], -1. * joy_msg.axes[4], 1. * joy_msg.axes[3]], nodes=ns)
            # orientation_tracking_gain.assign(cs.sqrt(1e5) if rotate else 0.)
        else:
            axis_x = keyboard.is_pressed('up') - keyboard.is_pressed('down')
            axis_y = keyboard.is_pressed('right') - keyboard.is_pressed('left')

            self.lip.rdot_ref.assign([alphaX * axis_x, alphaY * axis_y, 0], nodes=self.ns)
            # w_ref.assign([0, 0, 0], nodes=ns)
            # orientation_tracking_gain.assign(0.)

        self.lip.shiftContactConstraints()
        self.lip.setAction(motion, self.wpg)

        # solve
        tic()
        self.solver.solve()
        solution_time = toc()
        self.solution_time_vec.append(solution_time)
        self.solution_time_pub.publish(solution_time)
        self.solution = self.solver.getSolutionDict()

        c0_hist = dict()
        for i in range(0, self.lip.nc):
            c0_hist['c' + str(i)] = self.solution['c' + str(i)][:, 0]

        t = rospy.Time().now()
        utilities.SRBDTfBroadcaster(self.solution['r'][:, 0], np.array([0., 0., 0., 1.]), c0_hist, t)
        utilities.ZMPTfBroadcaster(self.solution['z'][:, 0], t)

        input = self.solution["u_opt"][:, 0]
        self.state = np.array(cs.DM(self.simulation_euler_integrator(self.state, input, self.solver.get_params_value(0))))

        rddot0 = self.lip.RDDOT(self.state, input, self.solver.get_params_value(0))
        fzmp = self.lip.m * (np.array([0., 0., 9.81]) + rddot0)
        viz.publishContactForce(t, fzmp, 'ZMP')
        for i in range(0, self.lip.nc):
            viz.publishPointTrj(self.solution["c" + str(i)], t, 'c' + str(i), "world", color=[0., 0., 1.])
        viz.SRBDViewer(np.eye(3), "SRB", t, self.lip.nc)  # TODO: should we use w_R_b * I * w_R_b.T?
        viz.publishPointTrj(self.solution["r"], t, "SRB", "world")
        viz.publishPointTrj(self.solution["z"], t, name="ZMP", frame="world", color=[0., 1., 1.], namespace="LIP")

        cc = dict()
        for i in range(0, self.lip.nc):
            cc[i] = self.solution["c" + str(i)][:, 0]

        return self.state, input, rddot0, fzmp

