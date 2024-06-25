import numpy as np

class steps_phase:
    def __init__(self, f, c, cdot, c_init_z, c_ref, cdot_switch, nodes, number_of_legs, contact_model, max_force, max_velocity):
        self.f = f
        self.c = c
        self.cdot = cdot
        self.c_ref = c_ref
        self.cdot_switch = cdot_switch

        self.number_of_legs = number_of_legs
        self.contact_model = contact_model

        self.nodes = nodes
        self.step_counter = 0

        #JUMP
        self.jump_c = []
        self.jump_cdot_bounds = []
        self.jump_f_bounds = []
        sin = 0.1 * np.sin(np.linspace(0, np.pi, 10))
        for k in range(0, 7):  # 7 nodes down
            self.jump_c.append(c_init_z)
            self.jump_cdot_bounds.append([0., 0., 0.])
            self.jump_f_bounds.append([max_force, max_force, max_force])
        for k in range(0, 8):  # 8 nodes jump
            self.jump_c.append(c_init_z + sin[k + 1])
            self.jump_cdot_bounds.append([max_velocity, max_velocity, max_velocity])
            self.jump_f_bounds.append([0., 0., 0.])
        for k in range(0, 7):  # 6 nodes down
            self.jump_c.append(c_init_z)
            self.jump_cdot_bounds.append([0., 0., 0.])
            self.jump_f_bounds.append([max_force, max_force, max_force])



        #NO STEP
        self.stance = []
        self.cdot_bounds = []
        self.f_bounds = []
        for k in range(0, nodes+1):
            self.stance.append([c_init_z])
            self.cdot_bounds.append([0., 0., 0.])
            self.f_bounds.append([max_force, max_force, max_force])


        #STEP
        sin = 0.1 * np.sin(np.linspace(0, np.pi, 10))
        #left step cycle
        self.l_cycle = []
        self.l_cdot_bounds = []
        self.l_cdot_switch = []
        self.l_f_bounds = []
        for k in range(0,2): # 2 nodes down
            self.l_cycle.append(c_init_z)
            self.l_cdot_bounds.append([0., 0., 0.])
            self.l_cdot_switch.append(1.)
            self.l_f_bounds.append([max_force, max_force, max_force])
        for k in range(0, 8):  # 8 nodes step
            self.l_cycle.append(c_init_z + sin[k + 1])
            self.l_cdot_bounds.append([max_velocity, max_velocity, max_velocity])
            self.l_cdot_switch.append(0.)
            self.l_f_bounds.append([0., 0., 0.])
        for k in range(0, 2):  # 2 nodes down
            self.l_cycle.append(c_init_z)
            self.l_cdot_bounds.append([0., 0., 0.])
            self.l_cdot_switch.append(1.)
            self.l_f_bounds.append([max_force, max_force, max_force])
        for k in range(0, 8):  # 8 nodes down (other step)
            self.l_cycle.append(c_init_z)
            self.l_cdot_bounds.append([0., 0., 0.])
            self.l_cdot_switch.append(1.)
            self.l_f_bounds.append([max_force, max_force, max_force])
        self.l_cycle.append(c_init_z) # last node down
        self.l_cdot_bounds.append([0., 0., 0.])
        self.l_cdot_switch.append(1.)
        self.l_f_bounds.append([max_force, max_force, max_force])

        # right step cycle
        self.r_cycle = []
        self.r_cdot_bounds = []
        self.r_cdot_switch = []
        self.r_f_bounds = []
        for k in range(0, 2):  # 2 nodes down
            self.r_cycle.append(c_init_z)
            self.r_cdot_bounds.append([0., 0., 0.])
            self.r_cdot_switch.append(1.)
            self.r_f_bounds.append([max_force, max_force, max_force])
        for k in range(0, 8):  # 8 nodes down (other step)
            self.r_cycle.append(c_init_z)
            self.r_cdot_bounds.append([0., 0., 0.])
            self.r_cdot_switch.append(1.)
            self.r_f_bounds.append([max_force, max_force, max_force])
        for k in range(0, 2):  # 2 nodes down
            self.r_cycle.append(c_init_z)
            self.r_cdot_bounds.append([0., 0., 0.])
            self.r_cdot_switch.append(1.)
            self.r_f_bounds.append([max_force, max_force, max_force])
        for k in range(0, 8):  # 8 nodes step
            self.r_cycle.append(c_init_z + sin[k + 1])
            self.r_cdot_bounds.append([max_velocity, max_velocity, max_velocity])
            self.r_cdot_switch.append(0.)
            self.r_f_bounds.append([0., 0., 0.])
        self.r_cycle.append(c_init_z)  # last node down
        self.r_cdot_bounds.append([0., 0., 0.])
        self.r_cdot_switch.append(1.)
        self.r_f_bounds.append([max_force, max_force, max_force])

        self.action = ""

    def set(self, action):
        t = self.nodes - self.step_counter

        for k in range(max(t, 0), self.nodes + 1):
            ref_id = (k - t)%self.nodes

            if(ref_id == 0):
                self.action = action

            if self.action == "trot":
                for i in [0, 3]:
                    self.c_ref[i].assign(self.l_cycle[ref_id], nodes = k)
                    self.cdot[i].setBounds(-1.*np.array(self.l_cdot_bounds[ref_id]), np.array(self.l_cdot_bounds[ref_id]), nodes=k)
                    self.cdot_switch[i].assign(self.l_cdot_switch[ref_id])
                    if k < self.nodes:
                        self.f[i].setBounds(-1.*np.array(self.l_f_bounds[ref_id]), np.array(self.l_f_bounds[ref_id]), nodes=k)
                for i in [1, 2]:
                    self.c_ref[i].assign(self.r_cycle[ref_id], nodes = k)
                    self.cdot[i].setBounds(-1.*np.array(self.r_cdot_bounds[ref_id]), np.array(self.r_cdot_bounds[ref_id]), nodes=k)
                    self.cdot_switch[i].assign(self.r_cdot_switch[ref_id])
                    if k < self.nodes:
                        self.f[i].setBounds(-1.*np.array(self.r_f_bounds[ref_id]), np.array(self.r_f_bounds[ref_id]), nodes=k)

            elif self.action == "step":
                for i in range(0, self.contact_model):
                    self.c_ref[i].assign(self.l_cycle[ref_id], nodes = k)
                    self.cdot[i].setBounds(-1.*np.array(self.l_cdot_bounds[ref_id]), np.array(self.l_cdot_bounds[ref_id]), nodes=k)
                    self.cdot_switch[i].assign(self.l_cdot_switch[ref_id])
                    if k < self.nodes:
                        self.f[i].setBounds(-1.*np.array(self.l_f_bounds[ref_id]), np.array(self.l_f_bounds[ref_id]), nodes=k)
                for i in range(self.contact_model, self.contact_model * self.number_of_legs):
                    self.c_ref[i].assign(self.r_cycle[ref_id], nodes = k)
                    self.cdot[i].setBounds(-1.*np.array(self.r_cdot_bounds[ref_id]), np.array(self.r_cdot_bounds[ref_id]), nodes=k)
                    self.cdot_switch[i].assign(self.r_cdot_switch[ref_id])
                    if k < self.nodes:
                        self.f[i].setBounds(-1.*np.array(self.r_f_bounds[ref_id]), np.array(self.r_f_bounds[ref_id]), nodes=k)

            elif self.action == "jump":
                for i in range(0, len(self.c)):
                    self.c_ref[i].assign(self.jump_c[ref_id], nodes = k)
                    self.cdot[i].setBounds(-1. * np.array(self.jump_cdot_bounds[ref_id]), np.array(self.jump_cdot_bounds[ref_id]), nodes=k)
                    if k < self.nodes:
                        self.f[i].setBounds(-1. * np.array(self.jump_f_bounds[ref_id]), np.array(self.jump_f_bounds[ref_id]), nodes=k)

            else:
                for i in range(0, len(self.c)):
                    self.c_ref[i].assign(self.stance[ref_id], nodes=k)
                    self.cdot[i].setBounds(-1. * np.array(self.cdot_bounds[ref_id]), np.array(self.cdot_bounds[ref_id]), nodes=k)
                    if k < self.nodes:
                        self.f[i].setBounds(-1. * np.array(self.f_bounds[ref_id]), np.array(self.f_bounds[ref_id]), nodes=k)

        self.step_counter += 1
