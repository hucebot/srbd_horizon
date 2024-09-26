import tf
import numpy as np
import casadi as cs
from scipy.spatial.transform import Rotation as R

def SRBDTfBroadcaster(r, o, c_dict, t):
    br = tf.TransformBroadcaster()
    br.sendTransform(r,o,t,"SRB","world")
    for key, val in c_dict.items():
        br.sendTransform(val, [0., 0., 0., 1.], t, key, "world")

def ZMPTfBroadcaster(zmp, t):
    br = tf.TransformBroadcaster()
    br.sendTransform(zmp, [0., 0., 0., 1.], t, "ZMP", "world")

def setWorld(frame, kindyn, q, base_link="base_link"):
    FRAME = kindyn.fk(frame)
    w_p_f = FRAME(q=q)['ee_pos']
    w_r_f = FRAME(q=q)['ee_rot']
    w_T_f = np.identity(4)
    w_T_f[0:3, 0:3] = w_r_f
    w_T_f[0:3, 3] = cs.transpose(w_p_f)

    BASE_LINK = kindyn.fk(base_link)
    w_p_bl = BASE_LINK(q=q)['ee_pos']
    w_r_bl = BASE_LINK(q=q)['ee_rot']
    w_T_bl = np.identity(4)
    w_T_bl[0:3, 0:3] = w_r_bl
    w_T_bl[0:3, 3] = cs.transpose(w_p_bl)

    w_T_bl_new = np.dot(np.linalg.inv(w_T_f), w_T_bl)

    rho = R.from_matrix(w_T_bl_new[0:3, 0:3]).as_quat()

    q[0:3] = w_T_bl_new[0:3, 3]
    q[3:7] = rho

def quat_inverse(q):
    p = q
    p[0:3] = -p[0:3]
    return p

def normalize_quaternion_part_horizon(q, ns):
    for n in range(ns+1):
        q[3:7, n] /= np.linalg.norm(q[3:7, n], ord=2)
    return q
