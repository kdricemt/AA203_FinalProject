import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def twobody_rates(s, t, u, mu):
    '''
    
    '''
    r = s[:3]
    v = s[3:] + u
    R = np.linalg.norm(r)
    a = -mu*r/R**3 
    dydt = np.hstack([v, a])

    return dydt

def unit_vector(v):
    return v/np.linalg.norm(v)


def eci2rsw(r_ECI,v_ECI):
    # Covert from ECI to LVLH
    rvec = unit_vector(r_ECI)
    # cross-track
    wvec = unit_vector(np.cross(r_ECI,v_ECI))
    # along-track
    svec = unit_vector(np.cross(wvec,rvec))

    # transformation matrix (ECI to RSW)
    T = np.vstack([rvec, svec, wvec])

    # position anc velocity
    rRSW = T @ r_ECI
    vRSW = T @ v_ECI

    return T, rRSW, vRSW


def eci_to_lvlh(r_target,v_target,r_chaise,v_chaise):
    # calculate the relative position and velocity in 
    # the LVLH frame
    # Coordinates
    #   - LVLH frame 
    #      - R: +radial direction
    #      - S: +velocity along-track
    #      - W: +angular momentum vector (cross-track)

    r_target_norm  = np.linalg.norm(r_target)
    r_chaise_norm  = np.linalg.norm(r_chaise)
    v_target_norm  = np.linalg.norm(v_target)

    # convert chaiser to ECI -> RSW  
    M_ECI2RSW, dum1, dum2 = eci2rsw(r_target,v_target)
    n_target = v_target_norm/r_target_norm

    # show relative state in RSW (LVLH) coordinate
    r_rho_RSW  = M_ECI2RSW @ (r_chaise - r_target).reshape((3,1))
    v_rho_RSW = M_ECI2RSW @ (v_chaise - v_target).reshape((3,1)) - np.array([-n_target * r_rho_RSW[1], n_target.*r_rho_RSW[0], 0]).reshape((3,1))

    # Convert from LVLH to orbital frame
    r_orbital = np.array([r_rho_RSW[1], -r_rho_RSW[2], -r_rho_RSW[0]])
    v_orbital = np.array([v_rho_RSW[1], -v_rho_RSW[2], -v_rho_RSW[0]])

    return r_orbital, v_orbital


def cw_stm(X0, t, n):
    # X0 (6,): relative position and velocity in orbital frame
    # t: propagation time (s)
    # n: angular velocity (rad/s) of the chief

    s = np.sin(n*t)
    c = np.cos(n*t)
    
    r0 = X0[:3]
    v0 = X0[3:]
    
    Phi_rr = np.array([[4 - 3*c, 0, 0], 
                       [-6*n*t + 6*s, 1, 0],
                       [0, 0, c]])
           
    Phi_rv = np.array([[s/n, 2/n - 2*c/n, 0],
                     [-2/n + 2*c/n, 4*s/n - 3*t, 0],
                     [0, 0, s/n]])
           
    Phi_vr = np.array([[3*n*s, 0, 0],
                       [-6*n + 6*n*c, 0, 0], 
                       [0, 0, -n*s]])
          
    Phi_vv = np.array([[c, 2*s, 0],
                       [-2*s, -3+4*c , 0],
                       [0 , 0 , c]])
          
    Phi = np.vstack([np.hstack([Phi_rr, Phi_rv]), np.hstack([Phi_vr, Phi_vv])])
    
    x0vec = X0.reshape((6,1))
    r_t  = Phi[:3,:] @ x0vec
    v_t  = Phi[3:,:] @ x0vec
   
    x_t = np.hstack([r_t, v_t])

    return x_t, Phi

def cw_ode(s, t, u, d):
    A = 
    ds = A @ s.reshape(6,1) + u + d

    return ds



def zero_control(t, s):
    return np.zeros(3)


def propagate_a(x0, tspan, controller, mu):
    lent = tspan.size
    n = x0.size
    s = np.zeros((lent,n))
    s[0,:] = x0
    for k in range(lent-1):
        t = tspan[k]
        u = controller(t, s[k])
        s[k+1, :] = odeint(twobody_rates, s[k,:], tspan[k:k+2], (u, mu))[1]

    return s

def propagate_relative_cw(x0, x0_ref, tspan, c)


def plot_orbit(s):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(s[:,0], s[:,1], s[:,2])

