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

        self.step_duration = 0.5
        self.dt = 0.05
        self.ss_share = 0.8
        self.ds_share = 0.2
        self.step_nodes = int(self.step_duration / self.dt)

        # generate step cycle
        ss_duration = int(self.ss_share * self.step_nodes)
        ds_duration = int(self.ds_share * self.step_nodes)
        sin = 0.1 * np.sin(np.linspace(0, np.pi, ))
        
        #left step cycle
        self.l_cycle = []
        self.l_cdot_switch = []
        for k in range(0, ds_duration):
            self.l_cycle.append(c_init_z)
            self.l_cdot_switch.append(1.)
        for k in range(0, ss_duration):
            self.l_cycle.append(c_init_z + sin[k + 1])
            self.l_cdot_switch.append(0.)
        for k in range(0, ds_duration):
            self.l_cycle.append(c_init_z)
            self.l_cdot_switch.append(1.)
        for k in range(0, ss_duration):
            self.l_cycle.append(c_init_z)
            self.l_cdot_switch.append(1.)
        self.l_cycle.append(c_init_z)
        self.l_cdot_switch.append(1.)

        # right step cycle
        self.r_cycle = []
        self.r_cdot_switch = []
        for k in range(0, ds_duration):
            self.r_cycle.append(c_init_z)
            self.r_cdot_switch.append(1.)
        for k in range(0, ss_duration):
            self.r_cycle.append(c_init_z)
            self.r_cdot_switch.append(1.)
        for k in range(0, ds_duration):
            self.r_cycle.append(c_init_z)
            self.r_cdot_switch.append(1.)
        for k in range(0, ss_duration):
            self.r_cycle.append(c_init_z + sin[k + 1])
            self.r_cdot_switch.append(0.)
        self.r_cycle.append(c_init_z)
        self.r_cdot_switch.append(1.)

        self.action = ""

    def set(self, action):

        self.action = action
        ref_id = self.step_counter % (2 * self.step_nodes)

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
        else: # stance
            self.w_ref.assign([0, 0., 0.], nodes=self.nodes)
            self.orientation_tracking_gain.assign(1e2, nodes=self.nodes)
            for i in range(0, len(self.c)):
                self.cdot_switch[i].assign(1., nodes=self.nodes)
                self.c_ref[i].assign(0., nodes=self.nodes)

        self.step_counter += 1


