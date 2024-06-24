import pyddp
from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.transcriptions import integrators
from typing import Dict
import casadi as cs
import numpy as np

class DDPSolver(Solver):
    def __init__(self, prb: Problem, opts: Dict) -> None:
        super().__init__(prb, opts=opts)
        self.prb = prb

        self.opts = opts
        self.max_iters = 100
        if "max_iters" in self.opts:
            self.max_iters = self.opts["max_iters"]
        self.alpha_0 = 1.0
        if "alpha_0" in self.opts:
            self.alpha_0 = self.opts["alpha_0"]
        self.alpha_converge_threshold = 1e-1
        if "alpha_converge_threshold" in self.opts:
            self.alpha_converge_threshold = self.opts["alpha_converge_threshold"]
        self.line_search_decrease_factor = 0.5
        if "line_search_decrease_factor" in self.opts:
            self.line_search_decrease_factor = self.opts["line_search_decrease_factor"]

        # generate problem to be solved
        self.var_container = self.prb.var_container
        self.fun_container = self.prb.function_container

        self.state_var = prb.getState().getVars()
        self.state_size = self.state_var.size()[0]
        self.input_var = prb.getInput().getVars()
        self.input_size = self.input_var.size()[0]
        self.param_var = prb.getParameters()

        #define discrete dynamics
        self.dae = dict()
        self.dae["x"] = cs.vertcat(self.state_var)
        self.dae["ode"] = self.prb.getDynamics()
        self.dae["p"] = cs.vertcat(self.input_var)
        self.dae["quad"] = 0.

        self.L_list = []
        self.f_list = []
        for n in range(0, prb.nodes-1):
            self.L_list.append(self.get_L(n))
            self.f_list.append(self.get_f(n))

        self.L_term = self.get_L_term(prb.nodes-1)

        self.param_values_list = list() # each element is a list of params per node
        for node in range(0, prb.nodes):
            self.param_values_list.append(self.get_params_value(node))

        self.ddp_solver = pyddp.DdpSolver(self.state_size, self.input_size, self.f_list, self.L_list, self.L_term,
                                          max_iters=self.max_iters, alpha_0=self.alpha_0,
                                          alpha_converge_threshold=self.alpha_converge_threshold,
                                          line_search_decrease_factor=self.line_search_decrease_factor)

    def solve(self) -> bool:
        # 1. update parameters
        for node in range(0, self.prb.nodes):
            self.param_values_list[node] = self.get_params_value(node)

        x, u = self.ddp_solver.solve(self.param_values_list)
        self.var_solution = self._createVarSolDict(x, u)
        self.var_solution['x_opt'] = x
        self.var_solution['u_opt'] = u

        return self.ddp_solver.is_converged()

    def set_u_warmstart(self, u):
        self.ddp_solver.set_u_warmstart(u)

    def getSolutionDict(self):
        return self.var_solution

    def setInitialState(self, x0):
        self.ddp_solver.set_initial_state(x0)

    def _createVarSolDict(self, x, u):
        #print(f"state_size: {self.state_size}")
        #print(f"input_size: {self.input_size}")
        #Each variable is a matrix in a dict, vars x nodes
        var_sol_dict = dict()
        var_size_acc = 0
        pos_x = 0
        pos_u = 0
        for var in self.prb.var_container.getVarList(offset=False):
            var_size_acc = var_size_acc + var.size()[0]
            if var_size_acc <= self.state_size:
                #print(f"var.size()[0]: {var.size()[0]}")
                #print(f"pos_x: {pos_x}")
                #print(f"var_size_acc: {var_size_acc}")
                val_sol_matrix = x[pos_x:var_size_acc, :]
                var_sol_dict[var.getName()] = val_sol_matrix
                pos_x = var_size_acc
            else:
                #print(f"var.size()[0]: {var.size()[0]}")
                #print(f"pos_u: {pos_u}")
                #print(f"var_size_acc: {var_size_acc - pos_x}")
                val_sol_matrix = u[pos_u:var_size_acc - pos_x, :]
                var_sol_dict[var.getName()] = val_sol_matrix
                pos_u = var_size_acc - pos_x
            #print(f"{var.getName()}: {var_sol_dict[var.getName()]}")

        return var_sol_dict

    def print_cost_info(self, cost):
        #print(dir(cost))

        print(f"{cost.getName()}:")
        print(f"    nodes: {cost.getNodes()}")
        print(f"    variables: {cost.getVariables()}")
        print(f"    parameters: {cost.getParameters()}")
        print(f"    dim: {cost.getDim()}")
        print(f"    function: {cost.getFunction()}")
        print(f"        in: {cost.getFunction().n_in()}")
        print(f"        out: {cost.getFunction().n_out()}")

    def get_params_value(self, node):
        param_values_at_node = list()
        for i_params in self.param_var.values():
            for i_param in i_params.getValues():
                param_values_at_node.append(i_param[node])
        return param_values_at_node

    def get_L(self, node):
        cost = 0
        for val in self.fun_container.getCost().values():
            if node in val.getNodes():
                #self.print_cost_info(val)
                vars = val.getVariables()
                pars = val.getParameters()
                simf = cs.sumsqr(val.getFunction()(*vars, *pars)) #note: we assume createResiduals functions!
                cost = cost + simf
        return cs.Function("L"+str(node), [cs.vertcat(self.state_var), cs.vertcat(self.input_var), cs.vcat(list(self.param_var.values()))], [cost])

    def get_L_term(self, node):
        cost = 0
        for val in self.fun_container.getCost().values():
            if node in val.getNodes():
                # self.print_cost_info(val)
                vars = val.getVariables()
                pars = val.getParameters()
                simf = cs.sumsqr(val.getFunction()(*vars, *pars))  # note: we assume createResiduals functions!
                cost = cost + simf
        return cs.Function("L" + str(node), [cs.vertcat(self.state_var),
                                             cs.vcat(list(self.param_var.values()))], [cost])

    def get_f(self, node):
        EULER = integrators.EULER(self.dae)
        return cs.Function("f" + str(node), [cs.vertcat(self.state_var), cs.vertcat(self.input_var)], [EULER(cs.vertcat(self.state_var), cs.vertcat(self.input_var), self.prb.getDt())[0]])