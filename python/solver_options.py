
def ipopt_offline_solver_options():
    i_opts = {
        'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 100,
        'ipopt.linear_solver': 'ma27',
        'ipopt.warm_start_init_point': 'no',
        'ipopt.fast_step_computation': 'no',
    }
    return i_opts

def ipopt_online_solver_options(max_iteration):
    opts = {
        'ipopt.accept_every_trial_step': 'yes',
        'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': max_iteration,
        'ipopt.linear_solver': 'ma27',
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.fast_step_computation': 'yes',
        'ipopt.print_level': 0,
        'ipopt.suppress_all_output': 'yes',
        'ipopt.sb': 'yes',
        'print_time': 0
    }
    return opts

def sqp_offline_solver_options(ns):
    i_opts = dict()
    i_opts['qp_solver'] = 'osqp'
    i_opts['warm_start_primal'] = True
    i_opts['warm_start_dual'] = True
    i_opts['gnsqp.eps_regularization'] = 1e-4
    i_opts['merit_derivative_tolerance'] = 1e-3
    i_opts['constraint_violation_tolerance'] = ns * 1e-3
    i_opts['osqp.polish'] = True  # without this
    i_opts['osqp.delta'] = 1e-6  # and this, it does not converge!
    i_opts['osqp.verbose'] = False
    i_opts['osqp.rho'] = 0.02
    i_opts['osqp.scaled_termination'] = False
    return i_opts

def sqp_online_solver_options(max_iterations):
    opts = {"gnsqp.max_iter": max_iterations,
            'gnsqp.osqp.scaled_termination': True,
            'gnsqp.eps_regularization': 1e-4,
            }
    return opts



