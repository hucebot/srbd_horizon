import casadi as cs
import numpy as np
from horizon import problem, variables
from horizon.utils import utils, kin_dyn, resampler_trajectory, mat_storer
from horizon.ros.replay_trajectory import *
from srbd_horizon import utilities


class LIPProblem:
    def __init__(self, namespace=""):
        self.namespace = namespace

    def createLIPProblem(self, ns, T, joint_init):
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
        contact_model = utilities.get_parm_from_paramserver("contact_model", self.namespace, 4)
        number_of_legs = utilities.get_parm_from_paramserver("number_of_legs", self.namespace, 2)
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

        if rospy.has_param("world_frame_link"):
            world_frame_link = rospy.get_param("world_frame_link")
            base_link_frame = "base_link"
            if rospy.has_param("base_link_frame"):
                base_link_frame = rospy.get_param("base_link_frame")
            utilities.setWorld(world_frame_link, kindyn, joint_init, base_link=base_link_frame)
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

        foot_frames = utilities.get_parm_from_paramserver("foot_frames", self.namespace, [])

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

        prb.createResidual("rel_pos_xy1", np.sqrt(rel_pos_gain) * (-c[0][0:2] + c[2][0:2] - d_initial_1[0:2]),
                           nodes=range(0, ns + 1))
        prb.createResidual("rel_pos_xy_2", np.sqrt(rel_pos_gain) * (-c[1][0:2] + c[3][0:2] - d_initial_2[0:2]),
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
        self.nodes = ns
        self.number_of_legs = number_of_legs

        self.step_counter = 0

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

    def shiftContactConstraints(self, end_node=None):
        if end_node is None:
            end_node = self.nodes + 1

        for j in range(1, end_node):
            for i in range(0, self.contact_model * self.number_of_legs):
                if self.cdot_switch is not None:
                    self.cdot_switch[i].assign(self.cdot_switch[i].getValues(nodes=j), nodes=j - 1)
                self.c_ref[i].assign(self.c_ref[i].getValues(nodes=j), nodes=j - 1)

    def shiftReferences(self, end_node=None):
        if end_node is None:
            end_node = self.nodes + 1

        for j in range(1, end_node):
            self.rdot_ref.assign(self.rdot_ref.getValues(nodes=j), nodes=j - 1)
            self.eta2_p.assign(self.eta2_p.getValues(nodes=j), nodes=j - 1)

    def assignReferences(self, rdot_ref_x, rdot_ref_y, rdot_ref_z):
        if (self.cdot_switch[0].getValues(self.nodes) == 0 and
                self.cdot_switch[1].getValues(self.nodes) == 0 and
                self.cdot_switch[2].getValues(self.nodes) == 0 and
                self.cdot_switch[3].getValues(self.nodes) == 0):
            self.eta2_p.assign(0., nodes=self.nodes)
        else:
            self.eta2_p.assign(self.eta2, nodes=self.nodes)

        self.rdot_ref.assign([rdot_ref_x, rdot_ref_y, rdot_ref_z], nodes=self.nodes)

    def setAction(self, action, plan):
        ref_id = self.step_counter % (2 * plan.step_nodes)

        if action == "walking":
            for i in range(0, self.contact_model):
                if self.cdot_switch is not None:
                    self.cdot_switch[i].assign(plan.l_cdot_switch[ref_id], nodes=self.nodes)
                self.c_ref[i].assign(plan.l_cycle[ref_id], nodes=self.nodes)

            for i in range(self.contact_model, self.contact_model * self.number_of_legs):
                if self.cdot_switch is not None:
                    self.cdot_switch[i].assign(plan.r_cdot_switch[ref_id], nodes=self.nodes)
                self.c_ref[i].assign(plan.r_cycle[ref_id], nodes=self.nodes)

        elif action == "jumping":
            for i in range(0, len(self.c)):
                if self.cdot_switch is not None:
                    self.cdot_switch[i].assign(plan.jump_cdot_switch[ref_id], nodes=self.nodes)
                self.c_ref[i].assign(plan.jump_c[ref_id], nodes=self.nodes)

        else: # stance
            for i in range(0, len(self.c)):
                if self.cdot_switch is not None:
                    self.cdot_switch[i].assign(1., nodes=self.nodes)
                self.c_ref[i].assign(0., nodes=self.nodes)

        self.step_counter += 1
