import casadi as cs
import numpy as np

import horizon.utils.utils
from horizon import problem, variables
from horizon.utils import utils, kin_dyn, resampler_trajectory, mat_storer
from horizon.ros.replay_trajectory import *
from srbd_horizon import utilities
from horizon.transcriptions import integrators

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
        return lv2 - lv1

    def kinematicTransmissionPosition(self, problem, q, FK1, FK2):
        lp1 = FK1(q=q)['ee_pos']
        lp2 = FK2(q=q)['ee_pos']
        return lp2 - lp1

    """
    This contains the full body problem for the kangaroo robot including the dynamic part of the transmission model
    """
    def createFullBodyProblem(self, ns, T, include_transmission_forces):
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
        self.nx = self.nq + self.nv

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
        qdot.setBounds(-10. * lims, 10. * lims)
        #qdot.setBounds(0. * lims, 0. * lims, nodes=0)


        # create input
        qddot = prb.createInputVariable("qddot", kindyn.nv())
        qddot.setBounds(-10. * lims, 10. * lims)

        contact_model = utilities.get_parm_from_paramserver("contact_model", self.namespace, 4)
        number_of_legs = utilities.get_parm_from_paramserver("number_of_legs", self.namespace, 2)
        nc = number_of_legs * contact_model
        foot_frames = utilities.get_parm_from_paramserver("foot_frames", self.namespace, [])
        foot_soles = utilities.get_parm_from_paramserver("foot_soles", self.namespace, [])

        f = dict()
        ones3 = np.ones(3)
        for foot_frame in foot_frames:
            f[foot_frame] = prb.createInputVariable("f_" + foot_frame, 3)  # Contact i forces
            f[foot_frame].setBounds(-1e4 * ones3, 1e4 * ones3)

        left_actuation_lambda = None
        right_actuation_lambda = None
        if include_transmission_forces:
            left_actuation_lambda = prb.createInputVariable("left_actuation_lambda", 2)
            left_actuation_lambda.setBounds(-1e4 * np.ones(2), 1e4 * np.ones(2))
            right_actuation_lambda = prb.createInputVariable("right_actuation_lambda", 2)
            right_actuation_lambda.setBounds(-1e4 * np.ones(2), 1e4 * np.ones(2))

        self.nu = self.nv + nc * 3
        if include_transmission_forces:
            self.nu += 2 * number_of_legs

        print(f"State: {prb.getState().getVars()}")
        print(f"Input: {prb.getInput().getVars()}")

        # Formulate discrete time dynamics
        x = cs.vertcat(q, qdot)
        xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)
        prb.setDynamics(xdot)
        prb.setDt(T / ns)
        dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': 0}
        F_integrator = integrators.RK2(dae, opts=None)

        # Constraints
        #1. multiple shooting
        x_next = cs.vertcat(q.getVarOffset(+1), qdot.getVarOffset(+1))
        x_int = F_integrator(x=x, u=qddot, dt=T/ns)
        prb.createConstraint("multiple_shooting", x_next - x_int["f"], nodes=list(range(0, ns)),
                             bounds=dict(lb=np.zeros(kindyn.nv() + kindyn.nq()),
                                         ub=np.zeros(kindyn.nv() + kindyn.nq())))


        #2. Torque limits including underactuation (notice: it includes as well the torque limits for the transmission)
        transmission_frames_left_leg = ["leg_left_length_link", "leg_left_knee_lower_bearing"] # <-- kangaroo related
        transmission_frames_right_leg = ["leg_right_length_link", "leg_right_knee_lower_bearing"] # <-- kangaroo related

        LJ1 = kindyn.jacobian(transmission_frames_left_leg[0], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        LJ2 = kindyn.jacobian(transmission_frames_left_leg[1], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        RJ1 = kindyn.jacobian(transmission_frames_right_leg[0], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        RJ2 = kindyn.jacobian(transmission_frames_right_leg[1], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

        tau_transmission = np.zeros(self.nv)
        if include_transmission_forces:
            tau_transmission = (self.computeTransmissionLegTorques(q, LJ1, LJ2, left_actuation_lambda) +
                                self.computeTransmissionLegTorques(q, RJ1, RJ2, right_actuation_lambda))

        tau_min = -np.array(torque_lims)
        tau_max = np.array(torque_lims)
        force_scaling = 1.
        tau = kin_dyn.InverseDynamics(kindyn, foot_frames, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q, qdot, qddot,  f, tau_ext=tau_transmission, wrench_scaling=force_scaling)
        if include_transmission_forces:
            prb.createIntermediateConstraint("inverse_dynamics", tau, nodes=list(range(0, ns)), bounds=dict(lb=tau_min, ub=tau_max))
        else:
            prb.createIntermediateConstraint("inverse_dynamics", tau[0:6], nodes=list(range(0, ns)), bounds=dict(lb=tau_min[0:6], ub=tau_max[0:6]))

        #3. kinematic constraints for transmission
        LV1 = kindyn.frameVelocity(transmission_frames_left_leg[0], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        LV2 = kindyn.frameVelocity(transmission_frames_left_leg[1], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        RV1 = kindyn.frameVelocity(transmission_frames_right_leg[0], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        RV2 = kindyn.frameVelocity(transmission_frames_right_leg[1], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

        LFK1 = kindyn.fk(transmission_frames_left_leg[0])
        LFK2 = kindyn.fk(transmission_frames_left_leg[1])
        RFK1 = kindyn.fk(transmission_frames_right_leg[0])
        RFK2 = kindyn.fk(transmission_frames_right_leg[1])

        prb.createConstraint("kinematic_transmission_left_leg_vel",
                             self.kinematicTransmissionVelocity(prb, q, qdot, LV1, LV2)[[0, 2]])
        prb.createConstraint("kinematic_transmission_right_leg_vel",
                             self.kinematicTransmissionVelocity(prb, q, qdot, RV1, RV2)[[0, 2]])
        prb.createConstraint("kinematic_transmission_left_leg_pos",
                             self.kinematicTransmissionPosition(problem, q, LFK1, LFK2)[[0, 2]])
        prb.createConstraint("kinematic_transmission_right_leg_pos",
                             self.kinematicTransmissionPosition(problem, q, RFK1, RFK2)[[0, 2]])

        #4. kinematic constraints for the feet + reference
        c_ref = dict()
        c = dict()
        co = dict()
        initial_foot_position = dict()

        cdotxy_tracking_constraint = dict()
        for foot_frame in foot_soles:
            print(foot_frame)
            FK = kindyn.fk(foot_frame)
            c_foot_frame = FK(q=q)['ee_pos']
            c[foot_frame] = c_foot_frame
            c_init = FK(q=joint_init)['ee_pos']
            initial_foot_position[foot_frame] = c_init
            c_ref[foot_frame] = prb.createParameter("c_ref_" + foot_frame, 1)
            c_ref[foot_frame].assign(c_init[2], nodes=range(0, ns + 1))
            prb.createConstraint("cz_tracking_" + foot_frame, c_foot_frame[2] - c_ref[foot_frame])

            #todo: set parameters for feet rotation
            from scipy.spatial.transform import Rotation as R
            co_init = R.from_matrix(FK(q=joint_init)['ee_rot']).as_quat()
            co[foot_frame] = self.rot2Quat(FK(q=q)['ee_rot'])
            prb.createConstraint("co_tracking_" + foot_frame, co[foot_frame] - co_init)


            DFK = kindyn.frameVelocity(foot_frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
            cdot_linear = DFK(q=q, qdot=qdot)['ee_vel_linear']
            cdotxy_tracking_constraint[foot_frame] = prb.createConstraint("cdotxy_tracking_" + foot_frame, cdot_linear)
            #prb.createResidual("min_cdot_" + foot_frame, 1e-1 * cdot_linear[0:2])

            cdot_angular = DFK(q=q, qdot=qdot)['ee_vel_angular']
            prb.createResidual("min_cw_" + foot_frame, 1e-1 * cdot_angular)

            #DDFK = kindyn.frameAcceleration(foot_frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
            #cddot_linear = DDFK(q=q, qdot=qdot, qddot=qddot)['ee_acc_linear']
            #prb.createResidual("min_cddot_" + foot_frame, 1e-1 * cddot_linear, nodes=range(0, ns))

            #cddot_angular = DDFK(q=q, qdot=qdot, qddot=qddot)['ee_acc_angular']
            #prb.createResidual("min_cwdot_" + foot_frame, 1e-1 * cddot_angular, nodes=range(0, ns))




        for foot_frame in foot_frames:
            mu = 0.8  # friction coefficient
            R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame
            fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f[foot_frame], mu, R)
            friction_cone = prb.createIntermediateConstraint(f"{foot_frame}_friction_cone", fc, bounds=dict(lb=fc_lb, ub=fc_ub))
            l = - 1e8 * np.ones((friction_cone.getLowerBounds().shape[0],1))
            friction_cone.setLowerBounds(l)

        # Cost function
        #1. minimize inputs
        qddot_prev = qddot.getVarOffset(-1)
        prb.createResidual("min_qddot", np.sqrt(1e-4) * (qddot_prev - qddot)/(T/ns), nodes=list(range(1, ns)))

        for foot_frame in foot_frames:
            f_ref = np.array([0., 0., (kindyn.mass()/force_scaling) * 9.81 / 8.])
            prb.createResidual("min_f_"+foot_frame, np.sqrt(1e-3) * (f[foot_frame]-f_ref), nodes=list(range(0, ns)))
        # prb.createResidual("min_left_actuation_lambda", np.sqrt(1e-3) * left_actuation_lambda, nodes=list(range(0, ns)))
        # prb.createResidual("min_right_actuation_lambda", np.sqrt(1e-3) * right_actuation_lambda, nodes=list(range(0, ns)))
        #prb.createResidual("min_q", np.sqrt(1e-3) * (q - joint_init))
        #prb.createResidual("min_qdot", np.sqrt(1e-3) * qdot)

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
        #prb.createResidual("rz_tracking", np.sqrt(r_tracking_gain) * (r[2] - com[2]), nodes=range(1, ns + 1))
        rdot_tracking_gain = rospy.get_param("rdot_tracking_gain", 1e3)
        #prb.createResidual("rdot_tracking", np.sqrt(rdot_tracking_gain) * (rdot - rdot_ref), nodes=range(1, ns + 1))

        prb.createResidual("z_tracking", np.sqrt(r_tracking_gain) * (q[2] - joint_init[2]), nodes=range(1, ns + 1))
        prb.createResidual("v_tracking", np.sqrt(rdot_tracking_gain) * (qdot[0:3] - rdot_ref), nodes=range(1, ns + 1))

        orientation_tracking_gain = prb.createParameter('orientation_tracking_gain', 1)
        orientation_tracking_gain.assign(1e3)
        w_tracking_gain = rospy.get_param("w_tracking_gain", 1e2)

        oref = prb.createParameter("oref", 4)
        oref.assign(np.array(joint_init[3:7]), nodes=range(1, ns + 1))
        w_ref = horizon.utils.utils.quaternion_error(q[3], q[4], q[5], q[6], oref[0], oref[1], oref[2], oref[3])
        prb.createResidual("w_tracking", np.sqrt(w_tracking_gain) * (qdot[3:6] - w_ref), nodes=range(1, ns + 1))

        #3. Keep feet separated
        d_initial_1 = -(initial_foot_position[foot_soles[0]][0:2] - initial_foot_position[foot_soles[1]][0:2])
        prb.createResidual("relative_pos_feet", np.sqrt(1e3) * (-c[foot_soles[0]][0:2] + c[foot_soles[1]][0:2] - d_initial_1))

        self.include_transmission_forces = include_transmission_forces
        self.prb = prb
        self.f = f
        self.q = q
        self.qdot = qdot
        self.qddot = qddot
        self.left_actuation_lambda = left_actuation_lambda
        self.right_actuation_lambda = right_actuation_lambda
        self.nc = nc
        self.foot_frames = foot_frames
        self.foot_soles = foot_soles
        self.joint_init = joint_init
        self.m = kindyn.mass()
        self.kindyn = kindyn
        self.c = c
        self.initial_foot_position = initial_foot_position
        self.c_ref = c_ref
        self.orientation_tracking_gain = orientation_tracking_gain
        self.contact_model = contact_model
        self.rdot_ref = rdot_ref
        #self.cdot_switch = cdot_switch
        self.cdotxy_tracking_constraint = cdotxy_tracking_constraint

        self.nodes = ns
        self.number_of_legs = number_of_legs
        self.step_counter = 0

        self.force_scaling = force_scaling

        self.createsInternalDataStructures()
        self.o_ref= oref
        self.joint_init = joint_init
        self.od = cs.DM(joint_init[3:7])

    def rot2Quat(self, R):
        w = 0.5 * cs.sqrt(1. + R[0, 0] + R[1, 1] + R[2, 2])
        x = 0.5 * (R[2, 1] - R[1, 2]) / (4. * w)
        y = 0.5 * (R[0, 2] - R[2, 0]) / (4. * w)
        z = 0.5 * (R[1, 0] - R[0, 1]) / (4. * w)

        return cs.vertcat(x, y, z, w)

    def getNonDynamicConstraintList(self):
        ng = list()
        for n in range(0, self.prb.getNNodes()):
            ng.append(0)
            for fun in self.prb.function_container.getCnstr().values():
                if fun.getName() != "multiple_shooting" and n < fun.getImpl().size2():
                    ng[n] += fun.getImpl().size1()
        print(ng)
        return ng

    def createsInternalDataStructures(self):
        self._f = dict()
        self._c = dict()
        self._c_ref = dict()
        self._cdotxy_tracking_constraint = dict()

        k = 0
        for foot_frame in self.foot_soles:
            self._c[k] = self.c[foot_frame]
            self._c_ref[k] = self.c_ref[foot_frame]
            self._cdotxy_tracking_constraint[k] = self.cdotxy_tracking_constraint[foot_frame]
            k += 1

        k = 0
        for foot_frame in self.foot_frames:
            self._f[k] = self.f[foot_frame]
            k += 1

    def getStateInputMappingMatrices(self):
        n = self.nq + self.nv

        lambda_size = 0
        if self.include_transmission_forces:
            lambda_size = 4
        m = 3 * self.nc + self.nv + lambda_size
        N = self.ns

        state_mapping_matrix = np.zeros((n * (N + 1), (n + m) * N + n))
        input_mapping_matrix = np.zeros((m * N, (n + m) * N + n))

        for k in range(0, N+1):
            state_mapping_matrix[k*n:k*n+n, k*(n+m):k*(n+m)+n] = np.identity(n)
            if k < N:
                input_mapping_matrix[k*m:k*m+m, k*(n+m)+n:k*(n+m)+n+m] = np.identity(m)

        return state_mapping_matrix, input_mapping_matrix

    def getInitialState(self):
        return np.concatenate((self.joint_init, np.zeros(self.nv)), axis=0)

    def getStaticInput(self):
        if self.include_transmission_forces:
            f = [0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., 0., 0.] #<-- 8 contact forces and 4 constraint forces
        else:
            f = [0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8,
                 0., 0., (self.m/self.force_scaling) * 9.81 / 8]
        return np.concatenate((np.zeros(self.nv), f), axis=0)

    def getInitialGuess(self):
        var_list = list()
        NNodes = self.ns + 1
        for k in range(0, NNodes - 1):
            for var in self.prb.var_container.getVarList(offset=False):
                retriever = var.getInitialGuess()
                var_list.append(retriever[:, k])
        is_state = lambda x: x == NNodes
        for var in self.prb.var_container.getVarList(offset=False):
            retriever = var.getInitialGuess()
            if is_state(retriever.shape[1]):
                var_list.append(retriever[:, NNodes - 1])
        v = cs.vertcat(*var_list)

        #print(v.print_vector(False))
        return v

    def shiftContactConstraints(self, end_node=None):
        if end_node is None:
            end_node = self.nodes + 1

        for j in range(1, end_node):
            for i in range(0, self.number_of_legs):
                self._cdotxy_tracking_constraint[i].setBounds(
                        self._cdotxy_tracking_constraint[i].getLowerBounds(node=j),
                        self._cdotxy_tracking_constraint[i].getUpperBounds(node=j), nodes=j - 1)
                self._c_ref[i].assign(self._c_ref[i].getValues(nodes=j), nodes=j - 1)


            for i in range(0, self.contact_model * self.number_of_legs):
                if j < self.nodes:
                    l, u = self._f[i].getBounds(node=j)
                    self._f[i].setBounds(l, u, nodes=j - 1)

    def shiftReferences(self, end_node=None):
        if end_node is None:
            end_node = self.nodes + 1

        for j in range(1, end_node):
            self.rdot_ref.assign(self.rdot_ref.getValues(nodes=j), nodes=j - 1)
            self.o_ref.assign(self.o_ref.getValues(nodes=j), nodes=j - 1)
            self.orientation_tracking_gain.assign(self.orientation_tracking_gain.getValues(nodes=j), nodes=j - 1)


    def setAction(self, action, plan):
        ref_id = self.step_counter % (2 * plan.step_nodes)

        if action == "walking":
            self.orientation_tracking_gain.assign(1e2, nodes=self.nodes)

            self._c_ref[0].assign(plan.l_cycle[ref_id], nodes=self.nodes)
            l = -1e4 * np.ones(3)
            l[2] = plan.dl_cycle[ref_id]
            u = 1e4 * np.ones(3)
            u[2] = plan.dl_cycle[ref_id]
            self._cdotxy_tracking_constraint[0].setBounds((1. - plan.l_cdot_switch[ref_id]) * l,
                                                          (1. - plan.l_cdot_switch[ref_id]) *  u,
                                                          nodes=self.nodes)

            for i in range(0, self.contact_model):
                self._f[i].setBounds(plan.l_cdot_switch[ref_id] * -1e4 * np.ones(3),
                                     plan.l_cdot_switch[ref_id] * 1e4 * np.ones(3), nodes=self.nodes-1)

            self._c_ref[1].assign(plan.r_cycle[ref_id], nodes=self.nodes)
            l = -1e4 * np.ones(3)
            l[2] = plan.dr_cycle[ref_id]
            u = 1e4 * np.ones(3)
            u[2] = plan.dr_cycle[ref_id]
            self._cdotxy_tracking_constraint[1].setBounds((1. - plan.r_cdot_switch[ref_id]) * l,
                                                          (1. - plan.r_cdot_switch[ref_id]) * u,
                                                          nodes=self.nodes)

            for i in range(self.contact_model, self.contact_model * self.number_of_legs):
                self._f[i].setBounds(plan.r_cdot_switch[ref_id] * -1e4 * np.ones(3),
                                     plan.r_cdot_switch[ref_id] * 1e4 * np.ones(3), nodes=self.nodes-1)

        elif action == "jumping":
            self.orientation_tracking_gain.assign(0., nodes=self.nodes)
            for i in range(0, len(self.c)):
                self._f[i].setBounds(plan.jump_cdot_switch[ref_id] * np.ones(3),
                                    plan.jump_cdot_switch[ref_id] * np.ones(3), nodes=self.nodes - 1)
                self._cdotxy_tracking_constraint[i].setBounds((1. - plan.jump_cdot_switch[ref_id]) * -1e4 * np.ones(3),
                                                              (1. - plan.jump_cdot_switch[ref_id]) *  1e4 * np.ones(3), nodes=self.nodes)
                self._c_ref[i].assign(plan.jump_c[ref_id], nodes=self.nodes)

        else: # stance
            self.orientation_tracking_gain.assign(1e2, nodes=self.nodes)
            for i in range(0, len(self.c)):
                self._c_ref[i].assign(0., nodes=self.nodes)
                self._f[i].setBounds(-1e4 * np.ones(3), 1e4 * np.ones(3), nodes=self.nodes - 1)
                self._cdotxy_tracking_constraint[i].setBounds(0. * np.ones(3), 0. * np.ones(3), nodes=self.nodes)

        self.step_counter += 1

    def assignVWReferences(self, rdot_ref_x, rdot_ref_y, rdot_ref_z, w_ref_x, w_ref_y, w_ref_z):
        self.rdot_ref.assign([rdot_ref_x, rdot_ref_y, rdot_ref_z], nodes=self.nodes)

        qw = cs.DM([self.prb.getDt() * 0.5 * w_ref_x,
                    self.prb.getDt() * 0.5 * w_ref_y,
                    self.prb.getDt() * 0.5 * w_ref_z,
                    0.])

        quatdot = cs.vertcat(*horizon.utils.utils.quaterion_product(self.od, qw))

        self.od[0] += quatdot[0]
        self.od[1] += quatdot[1]
        self.od[2] += quatdot[2]
        self.od[3] += quatdot[3]

        self.o_ref.assign(self.od, nodes=self.nodes)

    def assignVQReferences(self, rdot_ref_x, rdot_ref_y, rdot_ref_z, qx_ref, qy_ref, qz_ref, qw_ref):
        self.rdot_ref.assign([rdot_ref_x, rdot_ref_y, rdot_ref_z], nodes=self.nodes)

        self.od[0] = qx_ref
        self.od[1] = qy_ref
        self.od[2] = qz_ref
        self.od[3] = qw_ref

        self.o_ref.assign(self.od, nodes=self.nodes)

    def assignReferences(self, rdot_ref_x, rdot_ref_y, rdot_ref_z):
        self.rdot_ref.assign([rdot_ref_x, rdot_ref_y, rdot_ref_z], nodes=self.nodes)
        self.od = cs.DM(self.joint_init[3:7])
        self.o_ref.assign(self.od, nodes=self.nodes)

