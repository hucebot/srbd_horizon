import numpy as np
import casadi as cs

def LIP_dynamics(m, f_dict, r, rddot, p_dict):
    """
    Returns Single Rigid Body Dynamics constraint
    Args:
        m: robot mass
        f_dict: dictionary of contact forces
        r: com position
        rddot: com acceleration
    Returns:
        constraint as a vertcat
        m * rddot_xy - sum(f_xy) = 0
        0 = sum((p_i - r) x f_i)
    """
    eq1 = m * rddot[:2]
    eq2 = rddot[2]
    eq3 = 0
    for i, f in f_dict.items():
        eq1 = eq1 - f[:2]
        eq3 = eq3 - cs.mtimes(cs.skew(p_dict[i] - r),  f)

    return cs.vertcat(eq1, eq2, eq3)