import numpy as np

class steps_phase:
    def __init__(self, f, c, cdot, c_init_z, c_ref, w_ref, orientation_tracking_gain, cdot_switch, nodes, number_of_legs, contact_model):
        self.f = f
        self.c = c
        self.cdot = cdot
        self.c_ref = c_ref
        self.cdot_switch = cdot_switch
        self.w_ref = w_ref
        self.orientation_tracking_gain = orientation_tracking_gain

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
        for k in range(0, 8):  # 8 nodes jump
            self.jump_c.append(c_init_z + sin[k + 1])
        for k in range(0, 7):  # 6 nodes down
            self.jump_c.append(c_init_z)



        #NO STEP
        self.stance = []
        for k in range(0, nodes+1):
            self.stance.append([c_init_z])


        ss_duration = 8
        ds_duration = 2

        #STEP
        sin = 0.1 * np.sin(np.linspace(0, np.pi, 10))
        #left step cycle
        self.l_cycle = []
        self.l_cdot_switch = []
        for k in range(0, ds_duration): # 2 nodes down
            self.l_cycle.append(c_init_z)
            self.l_cdot_switch.append(1.)
        for k in range(0, ss_duration):  # 8 nodes step
            self.l_cycle.append(c_init_z + sin[k + 1])
            self.l_cdot_switch.append(0.)
        for k in range(0, ds_duration):  # 2 nodes down
            self.l_cycle.append(c_init_z)
            self.l_cdot_switch.append(1.)
        for k in range(0, ss_duration):  # 8 nodes down (other step)
            self.l_cycle.append(c_init_z)
            self.l_cdot_switch.append(1.)
        self.l_cycle.append(c_init_z) # last node down
        self.l_cdot_switch.append(1.)

        # right step cycle
        self.r_cycle = []
        self.r_cdot_switch = []
        for k in range(0, ds_duration):  # 2 nodes down
            self.r_cycle.append(c_init_z)
            self.r_cdot_switch.append(1.)
        for k in range(0, ss_duration):  # 8 nodes down (other step)
            self.r_cycle.append(c_init_z)
            self.r_cdot_switch.append(1.)
        for k in range(0, ds_duration):  # 2 nodes down
            self.r_cycle.append(c_init_z)
            self.r_cdot_switch.append(1.)
        for k in range(0, ss_duration):  # 8 nodes step
            self.r_cycle.append(c_init_z + sin[k + 1])
            self.r_cdot_switch.append(0.)
        self.r_cycle.append(c_init_z)  # last node down
        self.r_cdot_switch.append(1.)

        self.action = ""

    def set(self, action):

        self.action = action
        ref_id = self.step_counter%self.nodes

        # shift contact plan back by one node
        for j in range(1, self.nodes+1):
            for i in range(0, self.contact_model * self.number_of_legs):
                self.cdot_switch[i].assign(self.cdot_switch[i].getValues(nodes=j), nodes=j-1)
                self.c_ref[i].assign(self.c_ref[i].getValues(nodes=j), nodes=j-1)

        # fill last node of the contact plan
        if self.action == "step":
            self.w_ref.assign([0, 0., 0.], nodes=self.nodes)
            self.orientation_tracking_gain.assign(1e2, nodes=self.nodes)
            for i in range(0, self.contact_model):
                self.cdot_switch[i].assign(self.l_cdot_switch[ref_id], nodes=self.nodes)
                self.c_ref[i].assign(self.l_cycle[ref_id], nodes=self.nodes)
            for i in range(self.contact_model, self.contact_model * self.number_of_legs):
                self.cdot_switch[i].assign(self.r_cdot_switch[ref_id], nodes=self.nodes)
                self.c_ref[i].assign(self.r_cycle[ref_id], nodes=self.nodes)
        elif self.action == "jump":
            self.w_ref.assign([0, 0., 0.], nodes=self.nodes)
            self.orientation_tracking_gain.assign(0., nodes=self.nodes)
            for i in range(0, len(self.c)):
                self.cdot_switch[i].assign(0., nodes=self.nodes)
                self.c_ref[i].assign(self.jump_c[ref_id], nodes=self.nodes)
        else: # stance
            self.w_ref.assign([0, 0., 0.], nodes=self.nodes)
            self.orientation_tracking_gain.assign(1e2, nodes=self.nodes)
            for i in range(0, len(self.c)):
                self.cdot_switch[i].assign(1., nodes=self.nodes)
                self.c_ref[i].assign(0., nodes=self.nodes)

        self.step_counter += 1


