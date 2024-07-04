import pyddp
from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.transcriptions import integrators
from horizon.variables import Parameter
from typing import Dict
import casadi as cs
import numpy as np


def parameterized_euler(ode, state, dt):
    return state + dt * ode
class DDPSolver(Solver):
    def __init__(self, prb: Problem, opts: Dict) -> None:
        super().__init__(prb, opts=opts)
        self.prb = prb
        self.ddp_opts = pyddp.DdpSolverOptions()

        self.opts = opts
        self.max_iters = 100
        if "max_iters" in self.opts:
            self.ddp_opts.max_iters = self.opts["max_iters"]
        self.alpha_0 = 1.0
        if "alpha_0" in self.opts:
            self.ddp_opts.alpha_0 = self.opts["alpha_0"]
        self.ddp_opts.alpha_converge_threshold = 1e-1
        if "alpha_converge_threshold" in self.opts:
            self.ddp_opts.alpha_converge_threshold = self.opts["alpha_converge_threshold"]
        self.line_search_decrease_factor = 0.5
        if "line_search_decrease_factor" in self.opts:
            self.ddp_opts.line_search_decrease_factor = self.opts["line_search_decrease_factor"]
        self.beta = 1e-4
        if "beta" in self.opts:
            self.ddp_opts.beta = self.opts["beta"]
        if "cost_reduction_ths" in self.opts:
            self.ddp_opts.cost_reduction_ths = self.opts["cost_reduction_ths"]
        if "mu0" in self.opts:
            self.ddp_opts.mu0 = self.opts["mu0"]

        # generate problem to be solved
        self.var_container = self.prb.var_container
        self.fun_container = self.prb.function_container

        # get constraints
        self.equality_constraints = []
        self.inequality_constraints = []
        for constr in self.fun_container.getCnstr().values():
            if self.is_equality_constraint(constr):
                self.equality_constraints.append(constr)
            else:
                self.inequality_constraints.append(constr)

        # recover problem size
        self.state_var = prb.getState().getVars()
        self.state_size = self.state_var.size()[0]
        self.input_var = prb.getInput().getVars()
        self.input_size = self.input_var.size()[0]
        self.param_var = prb.getParameters()

        # prepare variables bound parameters
        # nodes_array = np.zeros(prb.nodes).astype(int)
        # nodes_array[None] = 1 #wtf?
        # for var in self.var_container.getVarList(offset=False):
        #     par_lower = Parameter(var.getName() + "lower",
        #                           var.shape[0],
        #                           nodes_array,
        #                           casadi_type=cs.MX,
        #                           abstract_casadi_type=cs.SX)
        #     par_upper = Parameter(var.getName() + "upper",
        #                           var.shape[0],
        #                           nodes_array,
        #                           casadi_type=cs.MX,
        #                           abstract_casadi_type=cs.SX)
        #     self.param_var[var.getName() + "lower"] = par_lower
        #     self.param_var[var.getName() + "upper"] = par_upper

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
                                          self.ddp_opts)

    def solve(self) -> bool:
        # 1. update parameters
        for node in range(0, self.prb.nodes):
            self.param_values_list[node] = self.get_params_value(node)

        x, u = self.ddp_solver.solve(self.param_values_list)
        self.var_solution = self._createVarSolDict(x, u)
        self.var_solution['x_opt'] = x
        self.var_solution['u_opt'] = u

        return self.ddp_solver.is_converged()

    def is_equality_constraint(self, constr):
        upper = np.array(constr.getUpperBounds())
        lower = np.array(constr.getLowerBounds())
        return np.linalg.norm(upper-lower) <= 1e-6

    def set_u_warmstart(self, u):
        self.ddp_solver.set_u_warmstart(u)

    def set_x_warmstart(self, x):
        self.ddp_solver.set_x_warmstart(x)

    def getSolutionDict(self):
        return self.var_solution

    def setInitialState(self, x0):
        self.ddp_solver.set_initial_state(x0)

    def _createVarSolDict(self, x, u):
        #Each variable is a matrix in a dict, vars x nodes
        var_sol_dict = dict()
        var_size_acc = 0
        pos_x = 0
        pos_u = 0
        for var in self.prb.var_container.getVarList(offset=False):
            var_size_acc = var_size_acc + var.size()[0]
            if var_size_acc <= self.state_size:
                val_sol_matrix = x[pos_x:var_size_acc, :]
                var_sol_dict[var.getName()] = val_sol_matrix
                pos_x = var_size_acc
            else:
                val_sol_matrix = u[pos_u:var_size_acc - pos_x, :]
                var_sol_dict[var.getName()] = val_sol_matrix
                pos_u = var_size_acc - pos_x

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
        #for var in self.var_container.getVarList(offset=False):
            # if node < var.getLowerBounds().shape[1]:
            #     #self.param_var[var.getName() + "lower"].assign(np.full(var.shape, -1e6)) #var.getLowerBounds()[:,node])
            #     #self.param_var[var.getName() + "upper"].assign(np.full(var.shape, 1e6)) #var.getUpperBounds()[:,node])
            #     self.param_var[var.getName() + "lower"].assign(np.nan_to_num(var.getLowerBounds()[:,node], posinf=1e9, neginf=-1e9))
            #     self.param_var[var.getName() + "upper"].assign(np.nan_to_num(var.getUpperBounds()[:,node], posinf=1e9, neginf=-1e9))

        param_values_at_node = list()
        for i_params in self.param_var.values():
            for i_param in i_params.getValues():
                param_values_at_node.append(i_param[node])
        return param_values_at_node

    def get_L(self, node):
        cost = 0
        constraint_weight = 1e6
        exp_parameter = 6.0
        for val in self.fun_container.getCost().values():
            if node in val.getNodes():
                #self.print_cost_info(val)
                vars = val.getVariables()
                pars = val.getParameters()
                simf = cs.sumsqr(val.getFunction()(*vars, *pars)) #note: we assume createResiduals functions!
                cost = cost + simf
        # equality constraints included with a large weight
        for constr in self.equality_constraints:
            if node in constr.getNodes():
                vars = constr.getVariables()
                pars = constr.getParameters()
                simf = cs.sumsqr(constr.getFunction()(*vars, *pars))
                cost = cost + constraint_weight * simf
        # inequality constraints through barrier functions
        # for constr in self.inequality_constraints:
        #     if node in constr.getNodes():
        #         vars = constr.getVariables()
        #         pars = constr.getParameters()
        #         simf = cs.sum1(exp_parameter * cs.exp(constr.getFunction()(*vars, *pars)))
        #         cost = cost + simf
        # add coincident bounds as equality constraints, and different bounds as inequality constraints
        # for var in self.var_container.getVarList(offset=False):
        #     simf = cs.sum1(cs.exp(exp_parameter * (var - self.param_var[var.getName()+"upper"])))
        #     cost = cost + simf
        #     simf = cs.sum1(cs.exp(exp_parameter * (self.param_var[var.getName()+"lower"] - var)))
        #     cost = cost + simf
        return cs.Function("L"+str(node), 
                           [cs.vertcat(self.state_var), 
                            cs.vertcat(self.input_var), 
                            cs.vcat(list(self.param_var.values()))], 
                           [cost])

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
        state = cs.vertcat(self.state_var)
        input = cs.vertcat(self.input_var)
        params = cs.vcat(list(self.param_var.values()))
        return cs.Function("f" + str(node), [state, input, params], [parameterized_euler(self.dae["ode"], state, self.prb.getDt())])

