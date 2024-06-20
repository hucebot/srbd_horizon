import pyddp
from horizon.solvers import Solver
from horizon.problem import Problem
from typing import Dict
import casadi as cs

class DDPSolver(Solver):
    def __init__(self, prb: Problem, opts: Dict) -> None:
        super().__init__(prb, opts=opts)
        self.prb = prb

        # generate problem to be solved
        self.var_container = self.prb.var_container
        self.fun_container = self.prb.function_container

        #print(dir(prb))
        self.state_var = prb.getState().getVars()
        self.input_var = prb.getInput().getVars()
        self.param_var = prb.getParameters()
        #exit()

        self.L_vec = []
        self.f_vec = []
        for n in range(0, prb.nodes-1):
            self.L_vec.append(self.get_L(n))
            self.f_vec.append(self.get_f(n))

        self.L_term = self.get_L_term(prb.nodes-1)




    def solve(self) -> bool:
        return True

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
        return cs.Function("f" + str(node), [cs.vertcat(self.state_var), cs.vertcat(self.input_var)], [self.prb.getDynamics()])