import numpy as np

class steps_phase:
    def __init__(self, number_of_legs, contact_model, c_init_z):

        self.number_of_legs = number_of_legs
        self.contact_model = contact_model

        self.step_counter = 0

        self.step_duration = 0.5
        self.dt = 0.05
        self.ss_share = 0.8 # percentuage of the step duration in single support
        self.ds_share = 0.2 # percentuage of the step duration in double support
        self.step_nodes = int(self.step_duration / self.dt)


        # generate step cycle
        ss_duration = int(self.ss_share * self.step_nodes)
        ds_duration = int(self.ds_share * self.step_nodes)
        sin = 0.1 * np.sin(np.linspace(0, np.pi, ))

        # JUMP
        self.jump_c = []
        self.jump_cdot_switch = []
        for k in range(0, 8):  # 8 nodes down
            self.jump_c.append(c_init_z)
            self.jump_cdot_switch.append(1.)
        for k in range(0, 8):  # 8 nodes jump
            self.jump_c.append(c_init_z + sin[k + 1])
            self.jump_cdot_switch.append(0.)
        for k in range(0, 4):  # 4 nodes down
            self.jump_c.append(c_init_z)
            self.jump_cdot_switch.append(1.)

        # WALK
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





