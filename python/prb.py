import casadi as cs
from horizon import problem, variables

import rospy
import casadi as cs
import numpy as np
from horizon import problem, variables
from horizon.utils import utils, kin_dyn, resampler_trajectory, mat_storer
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.transcriptions import integrators
from horizon.solvers import solver
from horizon.ros.replay_trajectory import *

import utilities

def get_parm_from_paramserver(param, namespace, default):
    param_value = 0
    if rospy.has_param(namespace + "/" + param):
        param_value = rospy.get_param(namespace + "/" + param)
    else:
        param_value = rospy.get_param(param, default)
    print(f"{namespace}/{param}: {param_value}")
    return param_value

class FullBodyProblem:
    def __init__(self, namespace=""):
        self.namespace = namespace

    def computeTransmissionLegTorques(self, q, J1, J2, transmission_lambda):
        lj1 = J1(q=q)['J']
        lj2 = J2(q=q)['J']
        J = (type(lj1)).zeros(2, lj1.shape[1])
        J[0, :] = lj2[0, :] - lj1[0, :]
        J[1, :] = lj2[2, :] - lj1[2, :]
        return cs.mtimes(J.T, transmission_lambda)

    def kinematicTransmissionVelocity(self, problem, q, qdot, V1, V2):
        lv1 = V1(q=q, qdot=qdot)['ee_vel_linear']
        lv2 = V2(q=q, qdot=qdot)['ee_vel_linear']
        return cs.vcat([lv2[0] - lv1[0], lv2[2] - lv1[2]])

    def kinematicTransmissionPosition(self, problem, q, FK1, FK2):
        lp1 = FK1(q=q)['ee_pos']
        lp2 = FK2(q=q)['ee_pos']
        return lp2 - lp1


    def createFullBodyProblem(self, ns, T):
        prb = problem.Problem(ns, casadi_type=cs.SX)

        urdf = rospy.get_param("robot_description", "")
        kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

        joint_init = rospy.get_param("joint_init")
        torque_lims = rospy.get_param("torque_lims")

        FK1 = kindyn.fk("base_link")
        FK2 = kindyn.fk("left_sole_link")
        p1 = FK1(q=joint_init)['ee_pos']
        p2 = FK2(q=joint_init)['ee_pos']
        p = p1 - p2
        joint_init[0:3] = np.array(p).flatten()

        self.nq = kindyn.nq()
        self.nv = kindyn.nv()
        self.ns = ns

        # create state
        q = prb.createStateVariable("q", kindyn.nq())
        q_min = np.array(kindyn.q_min())
        #q_min[0:3] = -1e6 * np.ones(3)
        q_max = np.array(kindyn.q_max())
        #q_max[0:3] = -q_min[0:3]
        q.setBounds(q_min, q_max)
        q.setBounds(joint_init, joint_init, nodes=0)
        q.setInitialGuess(joint_init)

        qdot = prb.createStateVariable("qdot", kindyn.nv())
        lims = np.ones(kindyn.nv())
        qdot.setBounds(-100. * lims, 100. * lims)


        # create input
        qddot = prb.createInputVariable("qddot", kindyn.nv())
        #qddot.setBounds(-10000. * lims, 10000. * lims)

        contact_model = get_parm_from_paramserver("contact_model", self.namespace, 4)
        number_of_legs = get_parm_from_paramserver("number_of_legs", self.namespace, 2)
        nc = number_of_legs * contact_model
        foot_frames = get_parm_from_paramserver("foot_frames", self.namespace, [])

        f = dict()
        ones3 = np.ones(3)
        for foot_frame in foot_frames:
            f[foot_frame] = prb.createInputVariable("f_" + foot_frame, 3)  # Contact i forces
            f[foot_frame].setBounds(-1e4 * ones3, 1e4 * ones3)

        left_actuation_lambda = prb.createInputVariable("left_actuation_lambda", 2)
        left_actuation_lambda.setBounds(-1e4 * np.ones(2), 1e4 * np.ones(2))
        right_actuation_lambda = prb.createInputVariable("right_actuation_lambda", 2)
        right_actuation_lambda.setBounds(-1e4 * np.ones(2), 1e4 * np.ones(2))

        # Formulate discrete time dynamics
        x = cs.vertcat(q, qdot)
        xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)
        prb.setDynamics(xdot)
        prb.setDt(T / ns)
        dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': 0}
        F_integrator = integrators.RK4(dae, opts=None) #originally EULER

        # Constraints
        #1. multiple shooting
        qddot_prev = qddot.getVarOffset(-1)
        x_prev = cs.vertcat(q.getVarOffset(-1), qdot.getVarOffset(-1))
        x_int = F_integrator(x=x_prev, u=qddot_prev, dt=T/ns)
        prb.createConstraint("multiple_shooting", x_int["f"] - x, nodes=list(range(1, ns + 1)), bounds=dict(lb=np.zeros(kindyn.nv() + kindyn.nq()), ub=np.zeros(kindyn.nv() + kindyn.nq())))

        #2. Torque limits including underactuation (notice: it includes as well the torque limits for the transmission)
        transmission_frames_left_leg = ["leg_left_length_link", "leg_left_knee_lower_bearing"] # <-- kangaroo related
        transmission_frames_right_leg = ["leg_right_length_link", "leg_right_knee_lower_bearing"] # <-- kangaroo related

        LJ1 = kindyn.jacobian(transmission_frames_left_leg[0], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        LJ2 = kindyn.jacobian(transmission_frames_left_leg[1], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        RJ1 = kindyn.jacobian(transmission_frames_right_leg[0], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        RJ2 = kindyn.jacobian(transmission_frames_right_leg[1], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

        tau_transmission = self.computeTransmissionLegTorques(q, LJ1, LJ2, left_actuation_lambda) + self.computeTransmissionLegTorques(q, RJ1, RJ2, right_actuation_lambda)

        tau_min = -np.array(torque_lims)
        tau_max = np.array(torque_lims)
        tau = kin_dyn.InverseDynamics(kindyn, foot_frames, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q, qdot, qddot,  f, tau_transmission)
        prb.createConstraint("inverse_dynamics", tau, nodes=list(range(0, ns)), bounds=dict(lb=tau_min, ub=tau_max))

        #3. kinematic constraints for transmission
        LV1 = kindyn.frameVelocity(transmission_frames_left_leg[0], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        LV2 = kindyn.frameVelocity(transmission_frames_left_leg[1], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        RV1 = kindyn.frameVelocity(transmission_frames_right_leg[0], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        RV2 = kindyn.frameVelocity(transmission_frames_right_leg[1], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        prb.createConstraint("kinematic_transmission_left_leg", self.kinematicTransmissionVelocity(prb, q, qdot, LV1, LV2))
        prb.createConstraint("kinematic_transmission_right_leg", self.kinematicTransmissionVelocity(prb, q, qdot, RV1, RV2))

        LFK1 = kindyn.fk(transmission_frames_left_leg[0])
        LFK2 = kindyn.fk(transmission_frames_left_leg[1])
        RFK1 = kindyn.fk(transmission_frames_right_leg[0])
        RFK2 = kindyn.fk(transmission_frames_right_leg[1])
        prb.createConstraint("left_leg_closed_chain", self.kinematicTransmissionPosition(problem, q, LFK1, LFK2))
        prb.createConstraint("right_leg_closed_chain", self.kinematicTransmissionPosition(problem, q, RFK1, RFK2))

        #4. kinematic constraints for the feet + reference
        c_ref = dict()
        c = dict()
        cdot = dict()
        cdot_switch = dict()
        initial_foot_position = dict()
        force_switch_weight = rospy.get_param("force_switch_weight", 1e2)
        for foot_frame in foot_frames:
            FK = kindyn.fk(foot_frame)
            c_foot_frame = FK(q=q)['ee_pos']
            c[foot_frame] = c_foot_frame
            c_init = FK(q=joint_init)['ee_pos']
            initial_foot_position[foot_frame] = c_init
            c_ref[foot_frame] = prb.createParameter("c_ref_" + foot_frame, 1)
            c_ref[foot_frame].assign(c_init[2], nodes=range(0, ns + 1))
            prb.createConstraint("cz_tracking_" + foot_frame, c_foot_frame[2] - c_ref[foot_frame])

            cdot_switch[foot_frame] = prb.createParameter("cdot_switch_" + foot_frame, 1)
            cdot_switch[foot_frame].assign(1., nodes=range(0, ns + 1))
            DFK = kindyn.frameVelocity(foot_frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
            cdot_linear = DFK(q=q, qdot=qdot)['ee_vel_linear']
            cdot[foot_frame] = cdot_linear
            cdot_angular = DFK(q=q, qdot=qdot)['ee_vel_angular']
            prb.createConstraint(f"{foot_frame}_cdot_angular", cdot_angular)
            prb.createConstraint("cdotxy_tracking_" + foot_frame, cdot_switch[foot_frame] * cdot[foot_frame][0:2])
            prb.createResidual("min_cdot_" + foot_frame, 1e-1 * cdot[foot_frame])

            mu = 0.8  # friction coefficient
            R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame
            fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f[foot_frame], mu, R)
            prb.createIntermediateConstraint(f"{foot_frame}_friction_cone", fc, bounds=dict(lb=fc_lb, ub=fc_ub))
            #prb.createIntermediateConstraint("f_" + foot_frame + "_active", (1. - cdot_switch[foot_frame]) * f[foot_frame])


        # Cost function
        #1. minimize inputs
        prb.createResidual("min_qddot", np.sqrt(1e-3) * qddot, nodes=list(range(0, ns)))
        for foot_frame in foot_frames:
            prb.createResidual("min_f_"+foot_frame, np.sqrt(1e-2) * f[foot_frame], nodes=list(range(0, ns)))
        # prb.createResidual("min_left_actuation_lambda", np.sqrt(1e-3) * left_actuation_lambda, nodes=list(range(0, ns)))
        # prb.createResidual("min_right_actuation_lambda", np.sqrt(1e-3) * right_actuation_lambda, nodes=list(range(0, ns)))
        # prb.createResidual("min_q", np.sqrt(1e-3) * (q - joint_init))
        # prb.createResidual("min_qdot", np.sqrt(1e-3) * qdot)

        #2 rdot and omega tracking
        rdot_ref = prb.createParameter('rdot_ref', 3)
        w_ref = prb.createParameter('w_ref', 3)
        rdot_ref.assign([0., 0., 0.], nodes=range(1, ns + 1))
        w_ref.assign([0., 0., 0.], nodes=range(1, ns + 1))

        COM = kindyn.centerOfMass()
        com = COM(q=joint_init)['com']
        r = COM(q=q)['com']
        rdot = COM(q=q, v=qdot)['vcom']

        # create cost function terms
        r_tracking_gain = rospy.get_param("r_tracking_gain", 1e3)
        prb.createResidual("rz_tracking", np.sqrt(r_tracking_gain) * (r[2] - com[2]), nodes=range(1, ns + 1))
        oref = prb.createParameter("oref", 4)

        oref.assign(np.array([0., 0., 0., 1.]))
        oi = cs.vcat([-q[3], -q[4], -q[5], q[6]])
        quat_error = cs.vcat(utils.quaterion_product(oref, oi))

        orientation_tracking_gain = prb.createParameter('orientation_tracking_gain', 1)
        orientation_tracking_gain.assign(1e4)
        prb.createResidual("o_tracking_xyz", orientation_tracking_gain * quat_error[0:3], nodes=range(1, ns + 1))
        prb.createResidual("o_tracking_w", orientation_tracking_gain * (quat_error[3] - 1.), nodes=range(1, ns + 1))
        rdot_tracking_gain = rospy.get_param("rdot_tracking_gain", 1e4)
        prb.createResidual("rdot_tracking", np.sqrt(rdot_tracking_gain) * (rdot - rdot_ref), nodes=range(1, ns + 1))
        w_tracking_gain = rospy.get_param("w_tracking_gain", 1e2)
        prb.createResidual("w_tracking", np.sqrt(w_tracking_gain) * (qdot[3:6] - w_ref), nodes=range(1, ns + 1))

        #3. Keep feet separated
        d_initial_1 = -(initial_foot_position[foot_frames[0]][0:2] - initial_foot_position[foot_frames[4]][0:2])
        prb.createResidual("relative_pos_y_1_4", 1e2 * (-c[foot_frames[0]][1] + c[foot_frames[4]][1] - d_initial_1[1]))
        prb.createResidual("relative_pos_x_1_4", 1e2 * (-c[foot_frames[0]][0] + c[foot_frames[4]][0] - d_initial_1[0]))
        d_initial_2 = -(initial_foot_position[foot_frames[1]][0:2] - initial_foot_position[foot_frames[6]][0:2])
        prb.createResidual("relative_pos_y_3_6", 1e2 * (-c[foot_frames[3]][1] + c[foot_frames[7]][1] - d_initial_2[1]))
        prb.createResidual("relative_pos_x_3_6", 1e2 * (-c[foot_frames[3]][0] + c[foot_frames[7]][0] - d_initial_2[0]))



        self.prb = prb
        self.f = f
        self.q = q
        self.qdot = qdot
        self.qddot = qddot
        self.left_actuation_lambda = left_actuation_lambda
        self.right_actuation_lambda = right_actuation_lambda
        self.nc = nc
        self.foot_frames = foot_frames
        self.joint_init = joint_init
        self.m = kindyn.mass()
        self.kindyn = kindyn
        self.c = c
        self.cdot = cdot
        self.initial_foot_position = initial_foot_position
        self.c_ref = c_ref
        self.w_ref = w_ref
        self.orientation_tracking_gain = orientation_tracking_gain
        self.contact_model = contact_model
        self.rdot_ref = rdot_ref
        self.oref = oref
        self.cdot_switch = cdot_switch

    def getStateInputMappingMatrices(self):
        n = self.nq + self.nv
        m = 3 * self.nc + self.nv + 4
        N = self.ns

        state_mapping_matrix = np.zeros((n * (N + 1), (n + m) * N + n))
        input_mapping_matrix = np.zeros((m * N, (n + m) * N + n))

        state_mapping_matrix[0:n*(N+1), 0:n*(N+1)] = np.identity(n*(N+1))
        input_mapping_matrix[0:m*N, n*(N+1):n*(N+1)+m*N] = np.identity(m*N)

        return state_mapping_matrix, input_mapping_matrix

    def getInitialState(self):
        return np.concatenate((self.joint_init, np.zeros(self.nv)), axis=0)

    def getStaticInput(self):
        f = [0., 0., self.m * 9.81 / 8,
             0., 0., self.m * 9.81 / 8,
             0., 0., self.m * 9.81 / 8,
             0., 0., self.m * 9.81 / 8,
             0., 0., self.m * 9.81 / 8,
             0., 0., self.m * 9.81 / 8,
             0., 0., self.m * 9.81 / 8,
             0., 0., self.m * 9.81 / 8,
             0., 0., 0., 0.] #<-- 8 contact forces and 4 constraint forces
        return np.concatenate((np.zeros(self.nv), f), axis=0)

    def getInitialGuess(self):
        var_list = list()
        for var in self.prb.var_container.getVarList(offset=False):
            retriever = var.getInitialGuess()
            var_list.append(retriever.flatten(order='F'))

        v = cs.vertcat(*var_list)
        return v

class SRBDProblem:
    def __init__(self, namespace=""):
        self.namespace = namespace

    def createSRBDProblem(self, ns, T):
        prb = problem.Problem(ns, casadi_type=cs.SX)

        urdf = rospy.get_param("robot_description", "")
        if urdf == "":
            print("robot_description not loaded in param server!")
            exit()

        kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

        # create variables

        r = prb.createStateVariable("r", 3)  # com position
        o = prb.createStateVariable("o", 4)  # base orientation quaternion
        q = variables.Aggregate()  # position aggregate
        q.addVariable(r)
        q.addVariable(o)

        # contacts position
        contact_model = get_parm_from_paramserver("contact_model", self.namespace, 4)
        number_of_legs = get_parm_from_paramserver("number_of_legs", self.namespace, 2)
        nc = number_of_legs * contact_model

        c = dict()
        for i in range(0, nc):
            c[i] = prb.createStateVariable("c" + str(i), 3)  # Contact i position
            q.addVariable(c[i])

        # variables
        rdot = prb.createStateVariable("rdot", 3)  # com velocity
        w = prb.createStateVariable("w", 3)  # base vel
        qdot = variables.Aggregate()  # velocity aggregate
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
            cddot[i] = prb.createInputVariable("cddot" + str(i), 3)  # Contact i acc
            f[i] = prb.createInputVariable("f" + str(i), 3)  # Contact i forces

        # references
        rdot_ref = prb.createParameter('rdot_ref', 3)
        w_ref = prb.createParameter('w_ref', 3)

        rdot_ref.assign([0., 0., 0.], nodes=range(1, ns + 1))
        w_ref.assign([0., 0., 0.], nodes=range(1, ns + 1))

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
        rddot, wdot = kin_dyn.fSRBD(m / force_scaling, w_R_b * (I / force_scaling) * w_R_b.T, f, r, c, w)  # scaled forces

        self.RDDOT = cs.Function('rddot', [prb.getInput().getVars()], [rddot])
        self.WDOT = cs.Function('wdot', [prb.getState().getVars(), prb.getInput().getVars()], [wdot])

        qddot.addVariable(cs.vcat([rddot, wdot]))
        for i in range(0, nc):
            qddot.addVariable(cddot[i])
        xdot = utils.double_integrator_with_floating_base(q.getVars(), qdot.getVars(), qddot.getVars(),
                                                          base_velocity_reference_frame=cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        prb.setDynamics(xdot)
        prb.setDt(T / ns)
        transcription_method = rospy.get_param("transcription_method",
                                               'multiple_shooting')  # can choose between 'multiple_shooting' and 'direct_collocation'
        transcription_opts = dict(integrator='RK2')  # integrator used by the multiple_shooting

        # foot_frames parameters are used to retrieve initial position of the contacts given the initial pose of the robot.
        # note: the order of the contacts state/control variable is the order in which these contacts are set in the param server

        foot_frames = get_parm_from_paramserver("foot_frames", self.namespace, [])

        i = 0
        initial_foot_position = dict()
        for frame in foot_frames:
            FK = kindyn.fk(frame)
            p = FK(q=joint_init)['ee_pos']
            print(f"{frame}: {p}")
            # storing initial foot_position and setting as initial bound
            initial_foot_position[i] = p
            i = i + 1

        # initialize com state and com velocity
        COM = kindyn.centerOfMass()
        com = COM(q=joint_init)['com']

        # weights
        r_tracking_gain = rospy.get_param("r_tracking_gain", 1e3)
        orientation_tracking_gain = prb.createParameter('orientation_tracking_gain', 1)
        orientation_tracking_gain.assign(1e1)
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
            c_ref[i].assign(initial_foot_position[i][2], nodes=range(0, ns + 1))
            cdot_switch[i] = prb.createParameter("cdot_switch" + str(i), 1)
            cdot_switch[i].assign(1., nodes=range(0, ns + 1))

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
            # prb.createIntermediateConstraint(f"f{i}_friction_cone", fc, bounds=dict(lb=fc_lb, ub=fc_ub))

        for i in range(0, nc):
            prb.createConstraint("cz_tracking" + str(i), c[i][2] - c_ref[i])
            prb.createConstraint("cdotxy_tracking" + str(i), cdot_switch[i] * cdot[i][0:2])

        # create cost function terms
        prb.createResidual("rz_tracking", np.sqrt(r_tracking_gain) * (r[2] - com[2]), nodes=range(1, ns + 1))
        oref = prb.createParameter("oref", 4)

        oref.assign(np.array([0., 0., 0., 1.]))
        oi = cs.vcat([-o[0], -o[1], -o[2], o[3]])
        quat_error = cs.vcat(utils.quaterion_product(oref, oi))
        #print(oref)
        #print(quat_error)

        #oref.assign(utilities.quat_inverse(np.array([0., 0., 0., 1.])))
        #quat_error = cs.vcat(utils.quaterion_product(o, oref))

        prb.createResidual("o_tracking_xyz", orientation_tracking_gain * quat_error[0:3], nodes=range(1, ns + 1))
        prb.createResidual("o_tracking_w", orientation_tracking_gain * (quat_error[3] - 1.), nodes=range(1, ns + 1))
        prb.createResidual("rdot_tracking", np.sqrt(rdot_tracking_gain) * (rdot - rdot_ref), nodes=range(1, ns + 1))
        prb.createResidual("w_tracking", np.sqrt(w_tracking_gain) * (w - w_ref), nodes=range(1, ns + 1))
        prb.createResidual("rel_pos_y_1_4", np.sqrt(rel_pos_gain) * (-c[0][1] + c[2][1] - d_initial_1[1]),
                           nodes=range(1, ns + 1))
        prb.createResidual("rel_pos_x_1_4", np.sqrt(rel_pos_gain) * (-c[0][0] + c[2][0] - d_initial_1[0]),
                           nodes=range(1, ns + 1))
        prb.createResidual("rel_pos_y_3_6", np.sqrt(rel_pos_gain) * (-c[1][1] + c[3][1] - d_initial_2[1]),
                           nodes=range(1, ns + 1))
        prb.createResidual("rel_pos_x_3_6", np.sqrt(rel_pos_gain) * (-c[1][0] + c[3][0] - d_initial_2[0]),
                           nodes=range(1, ns + 1))
        prb.createResidual("min_qddot", np.sqrt(min_qddot_gain) * (qddot.getVars()), nodes=range(0, ns))
        for i in range(0, nc):
            prb.createResidual("min_f" + str(i), force_scaling * np.sqrt(min_f_gain) * f[i], nodes=range(0, ns))
            prb.createResidual("f" + str(i) + "_active", force_scaling * np.sqrt(force_switch_weight)
                               * (1. - cdot_switch[i]) * f[i], nodes=range(0, ns))

        self.prb = prb
        self.initial_foot_position = initial_foot_position
        self.com = com
        self.force_scaling = force_scaling
        self.m = m
        self.f = f
        self.c = c
        self.cdot = cdot
        self.c_ref = c_ref
        self.w_ref = w_ref
        self.orientation_tracking_gain = orientation_tracking_gain
        self.cdot_switch = cdot_switch
        self.contact_model = contact_model
        self.rdot_ref = rdot_ref
        self.oref = oref
        self.nc = nc
        self.I = I

    def getInitialState(self):
        return np.array([float(self.com[0]), float(self.com[1]), float(self.com[2]),
                                  0., 0., 0., 1.,
                                  float(self.initial_foot_position[0][0]), float(self.initial_foot_position[0][1]),
                                  float(self.initial_foot_position[0][2]),
                                  float(self.initial_foot_position[1][0]), float(self.initial_foot_position[1][1]),
                                  float(self.initial_foot_position[1][2]),
                                  float(self.initial_foot_position[2][0]), float(self.initial_foot_position[2][1]),
                                  float(self.initial_foot_position[2][2]),
                                  float(self.initial_foot_position[3][0]), float(self.initial_foot_position[3][1]),
                                  float(self.initial_foot_position[3][2]),
                                  0., 0., 0.,
                                  0., 0., 0.,
                                  0., 0., 0.,
                                  0., 0., 0.,
                                  0., 0., 0.,
                                  0., 0., 0.])

    def getStaticInput(self):
        return np.array([0., 0., 0., 0., 0., self.m * 9.81 / self.force_scaling / 4,
                         0., 0., 0., 0., 0., self.m * 9.81 / self.force_scaling / 4,
                         0., 0., 0., 0., 0., self.m * 9.81 / self.force_scaling / 4,
                         0., 0., 0., 0., 0., self.m * 9.81 / self.force_scaling / 4])

class LIPProblem:
    def __init__(self, namespace=""):
        self.namespace = namespace

    def createLIPProblem(self, ns, T):
        prb = problem.Problem(ns, casadi_type=cs.SX)

        urdf = rospy.get_param("robot_description", "")
        if urdf == "":
            print("robot_description not loaded in param server!")
            exit()

        kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

        # create variables

        r = prb.createStateVariable("r", 3)  # com position
        q = variables.Aggregate()  # position aggregate
        q.addVariable(r)

        # contacts position
        contact_model = get_parm_from_paramserver("contact_model", self.namespace, 4)
        number_of_legs = get_parm_from_paramserver("number_of_legs", self.namespace, 2)
        nc = number_of_legs * contact_model

        c = dict()
        for i in range(0, nc):
            c[i] = prb.createStateVariable("c" + str(i), 3)  # Contact i position
            q.addVariable(c[i])

        # variables
        rdot = prb.createStateVariable("rdot", 3)  # com velocity
        qdot = variables.Aggregate()  # velocity aggregate
        qdot.addVariable(rdot)

        # contacts velocity
        cdot = dict()
        for i in range(0, nc):
            cdot[i] = prb.createStateVariable("cdot" + str(i), 3)  # Contact i vel
            qdot.addVariable(cdot[i])

        # variable to collect all acceleration controls
        qddot = variables.Aggregate()

        z = prb.createInputVariable("z", 3)  # zmp position
        cddot = dict()
        for i in range(0, nc):
            cddot[i] = prb.createInputVariable("cddot" + str(i), 3)  # Contact i acc

        # references
        rdot_ref = prb.createParameter('rdot_ref', 3)
        rdot_ref.assign([0., 0., 0.], nodes=range(1, ns + 1))

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

        # initialize com state and com velocity
        COM = kindyn.centerOfMass()
        com = COM(q=joint_init)['com']

        m = kindyn.mass()
        eta2 = 9.81 / com[2]
        eta2_p = prb.createParameter("eta", 1)
        eta2_p.assign(eta2)
        lip_dynamics = eta2_p * (r - z) - cs.DM([0., 0., 9.81])
        rddot = lip_dynamics

        qddot.addVariable(rddot)
        cddots = variables.Aggregate()
        for i in range(0, nc):
            qddot.addVariable(cddot[i])
            cddots.addVariable(cddot[i])
        print(q.getVars())
        print(qdot.getVars())
        xdot = utils.double_integrator(q.getVars(), qdot.getVars(), qddot.getVars())
        prb.setDynamics(xdot)
        prb.setDt(T / ns)

        # foot_frames parameters are used to retrieve initial position of the contacts given the initial pose of the robot.
        # note: the order of the contacts state/control variable is the order in which these contacts are set in the param server

        foot_frames = get_parm_from_paramserver("foot_frames", self.namespace, [])

        i = 0
        initial_foot_position = dict()
        for frame in foot_frames:
            FK = kindyn.fk(frame)
            p = FK(q=joint_init)['ee_pos']
            print(f"{frame}: {p}")
            # storing initial foot_position and setting as initial bound
            initial_foot_position[i] = p
            i = i + 1



        # weights
        r_tracking_gain = rospy.get_param("r_tracking_gain", 1e5)
        rdot_tracking_gain = rospy.get_param("rdot_tracking_gain", 1e4)
        zmp_tracking_gain = rospy.get_param("zmp_tracking_gain", 1e5) #1e3 in double support
        rel_pos_gain = rospy.get_param("rel_position_gain", 1e4)
        min_cddot_gain = rospy.get_param("min_cddot_gain", 1e0)

        # fixme: where do these come from?
        d_initial_1 = -(initial_foot_position[0][0:2] - initial_foot_position[2][0:2])
        d_initial_2 = -(initial_foot_position[1][0:2] - initial_foot_position[3][0:2])
        
        # create contact reference and contact switch
        c_ref = dict()
        cdot_switch = dict()
        for i in range(0, nc):
            c_ref[i] = prb.createParameter("c_ref" + str(i), 1)
            c_ref[i].assign(initial_foot_position[i][2], nodes=range(0, ns + 1))
            cdot_switch[i] = prb.createParameter("cdot_switch" + str(i), 1)
            cdot_switch[i].assign(1., nodes=range(0, ns + 1))

        # contact position constraints
        if contact_model > 1:
            for i in range(1, contact_model):
                prb.createConstraint("relative_vel_left_" + str(i), cdot[0][0:2] - cdot[i][0:2])
            for i in range(contact_model + 1, 2 * contact_model):
                prb.createConstraint("relative_vel_right_" + str(i), cdot[contact_model][0:2] - cdot[i][0:2])

        for i in range(0, nc):
            prb.createConstraint("cz_tracking" + str(i), c[i][2] - c_ref[i])
            prb.createConstraint("cdotxy_tracking" + str(i), cdot_switch[i] * cdot[i][0:2])

        # create cost function terms
        prb.createResidual("rz_tracking", np.sqrt(r_tracking_gain) * (r[2] - com[2]), nodes=range(0, ns + 1))
        prb.createResidual("rxy_tracking", np.sqrt(r_tracking_gain) * (r[:2] - (c[0]+c[1]+c[2]+c[3])[:2] * 0.25), nodes=range(0, ns + 1))
        prb.createResidual("rdot_tracking", np.sqrt(rdot_tracking_gain) * (rdot - rdot_ref), nodes=range(0, ns + 1))

        prb.createResidual("zmp_tracking_xy", np.sqrt(zmp_tracking_gain) * (z[0:2] - (cdot_switch[0]*c[0][0:2] +
                                                                              cdot_switch[1]*c[1][0:2] + cdot_switch[2]*c[2][0:2] +
                                                                              cdot_switch[3]*c[3][0:2]) / (sum(cdot_switch.values()) + 1e-2)), nodes=range(0, ns))

        prb.createResidual("zmp_tracking_z", np.sqrt(1e0) * z[2], nodes=range(0, ns))

        prb.createResidual("rel_pos_y_1_4", np.sqrt(rel_pos_gain) * (-c[0][1] + c[2][1] - d_initial_1[1]),
                           nodes=range(0, ns + 1))
        prb.createResidual("rel_pos_x_1_4", np.sqrt(rel_pos_gain) * (-c[0][0] + c[2][0] - d_initial_1[0]),
                           nodes=range(0, ns + 1))
        prb.createResidual("rel_pos_y_3_6", np.sqrt(rel_pos_gain) * (-c[1][1] + c[3][1] - d_initial_2[1]),
                           nodes=range(0, ns + 1))
        prb.createResidual("rel_pos_x_3_6", np.sqrt(rel_pos_gain) * (-c[1][0] + c[3][0] - d_initial_2[0]),
                           nodes=range(0, ns + 1))
        prb.createResidual("min_cddot", np.sqrt(min_cddot_gain) * (cddots.getVars()), nodes=range(0, ns))

        self.prb = prb
        self.initial_foot_position = initial_foot_position
        self.com = com
        self.m = m
        self.c = c
        self.cdot = cdot
        self.c_ref = c_ref
        self.cdot_switch = cdot_switch
        self.contact_model = contact_model
        self.rdot_ref = rdot_ref
        self.nc = nc
        self.eta2 = eta2
        self.eta2_p = eta2_p
        self.RDDOT = cs.Function('rddot', [prb.getState().getVars(), prb.getInput().getVars(),
                                           cs.vcat(list(prb.getParameters().values()))], [rddot])

    def getInitialState(self):
        return np.array([float(self.com[0]), float(self.com[1]), float(self.com[2]),
                         float(self.initial_foot_position[0][0]), float(self.initial_foot_position[0][1]),
                         float(self.initial_foot_position[0][2]),
                         float(self.initial_foot_position[1][0]), float(self.initial_foot_position[1][1]),
                         float(self.initial_foot_position[1][2]),
                         float(self.initial_foot_position[2][0]), float(self.initial_foot_position[2][1]),
                         float(self.initial_foot_position[2][2]),
                         float(self.initial_foot_position[3][0]), float(self.initial_foot_position[3][1]),
                         float(self.initial_foot_position[3][2]),
                         0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.])

    def getStaticInput(self):
        return np.array([float(self.com[0]), float(self.com[1]), 0.,
                         0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.])