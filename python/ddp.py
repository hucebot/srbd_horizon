import pyddp
from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.functions import Cost, RecedingCost, Residual, RecedingResidual
from horizon.transcriptions import integrators
from horizon.variables import Parameter
from typing import Dict
import casadi as cs
import numpy as np
import pyddp
import pysqpgn


class MetaSolver(Solver):
    def __init__(self, prb: Problem, opts: Dict) -> None:
        super().__init__(prb, opts=opts)

        self.solvers = list()
        self.mapping_functions = list()
        self.meta_solver = pyddp.MetaSolver()

        self.state_var = prb.getState().getVars()
        self.state_size = self.state_var.size()[0]
    def add(self, DDPSolver, mapping_function):
        self.solvers.append(DDPSolver)
        self.mapping_functions.append(mapping_function)
        self.meta_solver.add(DDPSolver.ddp_solver, mapping_function)

    def addSQP(self, SQPSolver, mapping_function):
        self.solvers.append(SQPSolver)
        self.mapping_functions.append(mapping_function)
        self.meta_solver.add(SQPSolver.solver, mapping_function)

    def setInitialState(self, initial_state):
        self.meta_solver.set_initial_state(initial_state)
    def solve(self) -> bool:
        params = list()
        for solver in self.solvers:
            params.append(solver.update_params())

        self.X, self.U = self.meta_solver.solve(params)

        self.var_solution = self._createVarSolDict(self.solvers[0].prb, self.X[0], self.U[0], self.state_size)
        self.var_solution['x_opt'] = self.X[0]
        self.var_solution['u_opt'] = self.U[0]

    def _createVarSolDict(self, prb, x, u, state_size):
        #Each variable is a matrix in a dict, vars x nodes
        var_sol_dict = dict()
        var_size_acc = 0
        pos_x = 0
        pos_u = 0
        for var in prb.var_container.getVarList(offset=False):
            var_size_acc = var_size_acc + var.size()[0]
            if var_size_acc <= state_size:
                val_sol_matrix = x[pos_x:var_size_acc, :]
                var_sol_dict[var.getName()] = val_sol_matrix
                pos_x = var_size_acc
            else:
                val_sol_matrix = u[pos_u:var_size_acc - pos_x, :]
                var_sol_dict[var.getName()] = val_sol_matrix
                pos_u = var_size_acc - pos_x

        return var_sol_dict

    def getSolutionDict(self):
        return self.var_solution

    def getSolutionModel(self, i):
        var_solution = self._createVarSolDict(self.solvers[i].prb, self.X[i], self.U[i], self.solvers[i].state_size)
        var_solution['x_opt'] = self.X[i]
        var_solution['u_opt'] = self.U[i]
        return var_solution

    def setMaxIterations(self, max_iters):
        self.meta_solver.set_max_iters(max_iters)



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
        self.update_params()

        x, u = self.ddp_solver.solve(self.param_values_list)
        self.var_solution = self._createVarSolDict(x, u)
        self.var_solution['x_opt'] = x
        self.var_solution['u_opt'] = u

    def update_params(self):
        for node in range(0, self.prb.nodes):
            self.param_values_list[node] = self.get_params_value(node)
        return self.param_values_list
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


class SQPSolver(Solver):

    def __init__(self, prb: Problem, opts: Dict, qp_solver_plugin: str) -> None:

        filtered_opts = None
        if opts is not None:
            filtered_opts = {k[6:]: opts[k] for k in opts.keys() if k.startswith('gnsqp.')}

        super().__init__(prb, opts=filtered_opts)
        

        if qp_solver_plugin == 'osqp':
            if 'osqp.verbose' not in self.opts:
                self.opts['osqp.verbose'] = False

            if 'osqp.polish' not in self.opts:
                self.opts['osqp.polish'] = True

        self.prb = prb
        self.param_var = prb.getParameters()

        # generate problem to be solved
        self.var_container = self.prb.var_container
        self.fun_container = self.prb.function_container

        # recover problem size
        self.state_var = prb.getState().getVars()
        self.state_size = self.state_var.size()[0]
        self.input_var = prb.getInput().getVars()
        self.input_size = self.input_var.size()[0]
        self.param_var = prb.getParameters()

        # generate problem to be solver
        var_list = list()
        for k in range(0, prb.getNNodes()-1):
            for var in prb.var_container.getVarList(offset=False):
                var_list.append(var.getImpl()[:, k])
        is_state = lambda x: x == prb.getNNodes()
        for var in prb.var_container.getVarList(offset=False):
            if is_state(var.getImpl().size2()):
                var_list.append(var.getImpl()[:, prb.getNNodes()-1])
        w = cs.veccat(*var_list)
        #print(w.print_vector(False))

        fun_list = list()
        for n in range(0, prb.getNNodes()):
            for fun in prb.function_container.getCnstr().values():
                if n < fun.getImpl().size2():
                    fun_list.append(fun.getImpl()[:, n])
        g = cs.veccat(*fun_list)

        # fun_list = list()
        # for fun in prb.function_container.getCnstr().values():
        #     fun_list.append(fun.getImpl())
        # g = cs.veccat(*fun_list)

        # todo: residual, recedingResidual should be the same class
        # sqp only supports residuals, warn the user otherwise
        fun_list = list()
        for n in range(0, prb.getNNodes()):
            for fun in self.fun_container.getCost().values():
                if n < fun.getImpl().size2():
                    fun_to_append = fun.getImpl()[:, n]
                if fun_to_append is not None:
                    if type(fun) in (Cost, RecedingCost):
                        print('warning: sqp solver does not support costs that are not residuals')
                        fun_list.append(fun_to_append[:])
                    elif type(fun) in (Residual, RecedingResidual):
                        fun_list.append(fun_to_append[:])
                    else:
                        raise Exception('wrong type of function found in fun_container')

        f = cs.veccat(*fun_list)

        # build parameters
        par_list = list()
        for par in self.var_container.getParList(offset=False):
            par_list.append(par.getImpl())
        p = cs.veccat(*par_list)

        # create solver from prob
        F = cs.Function('f', [w, p], [f], ['w', 'p'], ['f'])
        G = cs.Function('g', [w, p], [g], ['w', 'p'], ['g'])

        # create solver
        print(self.opts)
        self.solver = pysqpgn.SQPGNSX('gnsqp', qp_solver_plugin, F, G, self.opts)

        self.solver.setStateSize(self.prb.getState().getVars().size()[0])
        self.solver.setInputSize(self.prb.getInput().getVars().size()[0])
        self.solver.setHorizonSize(self.prb.getNNodes()-1)

    def setStateInputMapping(self, state_mapping_matrix, input_mapping_matrix):
        self.solver.setStateInputMapping(state_mapping_matrix, input_mapping_matrix)

    def update_params(self):
        p = self._getParList()
        if p.size1() == 0:
            p = cs.DM([])
        #print(f"np.array(p, dtype=np.float64).flatten() {np.array(p, dtype=np.float64).flatten()}")
        #exit()
        return np.array(p, dtype=np.float64)

    def setInitialGuess(self, w_guess):
        self.solver.setInitialGuess(w_guess)

    def updateBounds(self, update_solver=True):
        #lbw = np.array(self._getVarList('lb'), dtype=np.float64).flatten()
        #ubw = np.array(self._getVarList('ub'), dtype=np.float64).flatten()


        lbw_list = list()
        ubw_list = list()
        for k in range(0, self.prb.getNNodes() - 1):
            for var in self.prb.var_container.getVarList(offset=False):
                lb = var.getLowerBounds()
                ub = var.getUpperBounds()
                lbw_list.append(lb[:, k])
                ubw_list.append(ub[:, k])
        is_state = lambda x: x == self.prb.getNNodes()
        for var in self.prb.var_container.getVarList(offset=False):
            lb = var.getLowerBounds()
            ub = var.getUpperBounds()
            if is_state(lb.shape[1]):
                lbw_list.append(lb[:, self.prb.getNNodes() - 1])
                ubw_list.append(ub[:, self.prb.getNNodes() - 1])

        lbw = np.concatenate(lbw_list)
        ubw = np.concatenate(ubw_list)

        if update_solver:
            self.solver.updateBounds(lbw, ubw)

        return lbw, ubw

    def updateConstraints(self, update_solver=True):
        #lbg = np.array(self._getFunList('lb'), dtype=np.float64).flatten()
        #ubg = np.array(self._getFunList('ub'), dtype=np.float64).flatten()

        lbg = np.zeros((self._getFunList('lb').size1(), 1))
        ubg = np.zeros((self._getFunList('ub').size1(), 1))

        s = 0
        for n in range(0, self.prb.getNNodes()):
            for cnstr in self.prb.function_container.getCnstr():
                l = self.prb.function_container.getCnstr()[cnstr].getLowerBounds()
                u = self.prb.function_container.getCnstr()[cnstr].getUpperBounds()
                if n < l.shape[1]:
                    lbg[s:s+l.shape[0], 0] = l[:, n]
                    ubg[s:s+u.shape[0], 0] = u[:, n]
                    s += l.shape[0]

        if update_solver:
            self.solver.updateConstraints(lbg, ubg)

        return lbg, ubg

    def createVarSolDict(self, sol):
        state_size = self.prb.getState().getVars().size()[0]
        input_size = self.prb.getInput().getVars().size()[0]

        self.x_opt = np.zeros((state_size, self.prb.getNNodes()))
        self.u_opt = np.zeros((input_size, self.prb.getNNodes()-1))

        for n in range(0, self.prb.getNNodes() - 1):
            node = sol["x"][n*(state_size + input_size):(n+1)*(state_size + input_size)]
            self.x_opt[:, n] = node[0:state_size]
            self.u_opt[:, n] = node[state_size:(state_size + input_size)]
        self.x_opt[:, -1] = sol["x"][-state_size:]

        pos = 0
        for state in self.prb.getState().var_list:
            name = state.getName()
            dim = state.getDim()
            self.var_solution[name] = np.zeros((dim, self.prb.getNNodes()))
            for n in range(0, self.prb.getNNodes()):
                self.var_solution[name][:, n] = self.x_opt[pos: pos + dim, n]
            pos += dim

        pos = 0
        for input in self.prb.getInput().var_list:
            name = input.getName()
            dim = input.getDim()
            self.var_solution[name] = np.zeros((dim, self.prb.getNNodes()-1))
            for n in range(0, self.prb.getNNodes()-1):
                self.var_solution[name][:, n] = self.u_opt[pos: pos + dim, n]
            pos += dim

    def solve(self) -> bool:

        # update lower/upper bounds of variables
        lbw, ubw = self.updateBounds(update_solver=False)

        # update lower/upper bounds of constraints
        lbg, ubg = self.updateConstraints(update_solver=False)

        # update parameters
        p = self._getParList()
        if p.size1() == 0:
            p = cs.DM([])
        p = np.array(p, dtype=np.float64).flatten()

        # solve
        sol = self.solver.solve(p=p, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        self.createVarSolDict(sol)

        # get solution dict
        #self.var_solution = self._createVarSolDict(sol)
        self.var_solution['x_opt'] = self.x_opt
        self.var_solution['u_opt'] = self.u_opt

        # get solution as state/input
        #self._createVarSolAsInOut(sol)

        # build dt_solution as an array
        #self._createDtSol()

        return True

    def getSolutionDict(self):
        return self.var_solution

    def getConstraintSolutionDict(self):
        return self.cnstr_solution

    def getDt(self):
        return self.dt_solution

    def setAlphaMin(self, alpha_min):
        self.solver.setAlphaMin(alpha_min)

    def getAlpha(self):
        return self.solver.getAlpha()

    def getBeta(self):
        return self.solver.getBeta()

    def setBeta(self, beta):
        self.solver.setBeta(beta)
