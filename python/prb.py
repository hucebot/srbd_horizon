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

class SRBDProblem:
    def __init__(self):
        None

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
        contact_model = rospy.get_param("contact_model", 4)
        number_of_legs = rospy.get_param("number_of_legs", 2)
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

        foot_frames = rospy.get_param("foot_frames")
        if len(foot_frames) == 0:
            print("foot_frames parameter is mandatory, exiting...")
            exit()
        if (len(foot_frames) != nc):
            print(f"foot frames number should match number of contacts! {len(foot_frames)} != {nc}")
            exit()
        print(f"foot_frames: {foot_frames}")

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
        oref.assign(utilities.quat_inverse(np.array([0., 0., 0., 1.])))
        quat_error = cs.vcat(utils.quaterion_product(o, oref))
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
    def __init__(self):
        None

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
        contact_model = rospy.get_param("contact_model", 4)
        number_of_legs = rospy.get_param("number_of_legs", 2)
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

        m = kindyn.mass()
        force_scaling = 1000.
        eta2 = 9.81 / 0.88
        lip_dynamics = eta2 * (r - z) - cs.DM([0., 0., 9.81])
        rddot = lip_dynamics

        #self.RDDOT = cs.Function('rddot', [prb.getInput().getVars()], [rddot])

        qddot.addVariable(rddot)
        for i in range(0, nc):
            qddot.addVariable(cddot[i])
        print(q.getVars())
        print(qdot.getVars())
        xdot = utils.double_integrator(q.getVars(), qdot.getVars(), qddot.getVars())
        prb.setDynamics(xdot)
        prb.setDt(T / ns)

        # foot_frames parameters are used to retrieve initial position of the contacts given the initial pose of the robot.
        # note: the order of the contacts state/control variable is the order in which these contacts are set in the param server

        foot_frames = rospy.get_param("foot_frames")
        if len(foot_frames) == 0:
            print("foot_frames parameter is mandatory, exiting...")
            exit()
        if (len(foot_frames) != nc):
            print(f"foot frames number should match number of contacts! {len(foot_frames)} != {nc}")
            exit()
        print(f"foot_frames: {foot_frames}")

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
        rdot_tracking_gain = rospy.get_param("rdot_tracking_gain", 1e4)
        zmp_tracking_gain = rospy.get_param("zmp_tracking_gain", 1e3)
        rel_pos_gain = rospy.get_param("rel_position_gain", 1e4)
        min_qddot_gain = rospy.get_param("min_qddot_gain", 1e0)

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
        prb.createResidual("rz_tracking", np.sqrt(r_tracking_gain) * (r[2] - com[2]), nodes=range(1, ns + 1))
        prb.createResidual("rxy_tracking", np.sqrt(r_tracking_gain) * (r[:2] - (c[0]+c[1]+c[2]+c[3])[:2] * 0.25), nodes=range(1, ns + 1))
        prb.createResidual("rdot_tracking", np.sqrt(rdot_tracking_gain) * (rdot - rdot_ref), nodes=range(1, ns + 1))
        prb.createResidual("zmp_tracking", np.sqrt(zmp_tracking_gain) * (z - (c[0]+c[1]+c[2]+c[3]) * 0.25), nodes=range(0, ns))
        prb.createResidual("rel_pos_y_1_4", np.sqrt(rel_pos_gain) * (-c[0][1] + c[2][1] - d_initial_1[1]),
                           nodes=range(1, ns + 1))
        prb.createResidual("rel_pos_x_1_4", np.sqrt(rel_pos_gain) * (-c[0][0] + c[2][0] - d_initial_1[0]),
                           nodes=range(1, ns + 1))
        prb.createResidual("rel_pos_y_3_6", np.sqrt(rel_pos_gain) * (-c[1][1] + c[3][1] - d_initial_2[1]),
                           nodes=range(1, ns + 1))
        prb.createResidual("rel_pos_x_3_6", np.sqrt(rel_pos_gain) * (-c[1][0] + c[3][0] - d_initial_2[0]),
                           nodes=range(1, ns + 1))
        prb.createResidual("min_qddot", np.sqrt(min_qddot_gain) * (qddot.getVars()), nodes=range(0, ns))

        self.prb = prb
        self.initial_foot_position = initial_foot_position
        self.com = com
        self.force_scaling = force_scaling
        self.m = m
        self.c = c
        self.cdot = cdot
        self.c_ref = c_ref
        self.cdot_switch = cdot_switch
        self.contact_model = contact_model
        self.rdot_ref = rdot_ref
        self.nc = nc

        print(prb.getState().getVars())
        print(prb.getInput().getVars())

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