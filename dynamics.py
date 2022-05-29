import numpy as np
from scipy.integrate import odeint
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pykep as pk
from pykep.core import par2ic, MU_EARTH, epoch_from_string
import PIL


def propagate_absolute_orbit(x0, tspan, controller, mu, earth_to_sun, T,
                             pflag=[True, True, True], update_parameter=True, debug=False):
    """
    Propagate absolute orbit of the satellite
    """
    lent = tspan.size
    n = x0.size
    s = np.zeros((lent, n))
    s[0,:] = x0
    for k in range(lent-1):
        t = tspan[k]
        u = controller(t, s[k])
        s[k+1, :] = odeint(absolute_dynamics, s[k,:], tspan[k:k+2],
                           (u, mu, earth_to_sun, T, pflag, update_parameter))[1]
        if debug:
            print("Step:{0:4d}/{1:4d} | x= {2:.2e}  y = {3:.2e}  z= {4:.2e}".format(k+1, lent, s[k+1,0], s[k+1,1], s[k+1,2]))

    return s

def propagate_relative_orbit(x0, tspan, controller, mu, earth_to_sun, T,
                             pflag=[True, True, True], update_parameter=True, debug=False):
    """
    Propagate relative orbit along with the absolute orbit of the reference
    """
    lent = tspan.size
    n = x0.size
    s = np.zeros((lent, n))
    s[0,:] = x0

    for k in range(lent-1):
        t = tspan[k]
        u = controller(t, s[k])

        s[k+1, :] = odeint(relative_dynamics, s[k,:], tspan[k:k+2],
                           (u, mu, earth_to_sun, T, pflag, update_parameter))[1]
        if debug:
            print("Step:{0:4d}/{1:4d} | x= {2:.2e}  y = {3:.2e}  z= {4:.2e}".format(k+1, lent, s[k+1,0], s[k+1,1], s[k+1,2]))

    return s


def unit_vector(v):
    """
    Compute unit vector
    """
    return v / jnp.linalg.norm(v)


def twobody_rates(s, t, u, mu):
    '''
    Compute the Acceleration for simple two body dynamics
    '''
    r = s[:3]
    v = s[3:6]
    R = np.linalg.norm(r)
    a = -mu*r/R**3 + u
    dydt = np.hstack([v, a])
    return dydt


def rotation_eci2rsw(r_ECI, v_ECI):
    # Covert from ECI to LVLH
    rvec = unit_vector(r_ECI)
    # cross-track
    wvec = unit_vector(jnp.cross(r_ECI,v_ECI))
    # along-track
    svec = unit_vector(jnp.cross(wvec,rvec))

    # transformation matrix (ECI to RSW)
    T = jnp.vstack([rvec, svec, wvec])

    return T

def absolute_dynamics(s, t, u, mu, earth_to_sun, T,
                      pflag=[True, True, True], update_parameter=True, debug=False):
    """
    Absolute Dynamics of the spacecraft
    """

    # Two body dynamics
    r = s[:3]
    R = jnp.linalg.norm(r)
    a_body = -mu*r/R**3

    gamma_srp, gamma_D, psi = s[6], s[7], s[8]

    # calculate perturbation
    if pflag[0]:
        a_srp = solar_radiation_pressure(gamma_srp, s[:3], earth_to_sun)
    else:
        a_srp = jnp.zeros(3)

    if pflag[1]:
        a_drag = drag_acceleration(gamma_D, s[:6])
    else:
        a_drag = jnp.zeros(3)

    # calculate perturbated control input
    if pflag[2] and jnp.any(u):
        u_p = thermal_fluttering(psi, u, earth_to_sun, s[:6], jnp.eye(3))  # No need to convert to LVLH -> pass eye matrix for eci2rsw
    else:
        u_p = u

    a_total = a_body + a_srp + a_drag + u_p

    if debug:
        print("Total Acceleration: {}".format(jnp.linalg.norm(a_total)))

    if update_parameter:
        sdot = jnp.hstack([s[3:6], a_total, parameter_dynamics(t, T)])
    else:
        sdot = jnp.hstack([s[3:6], a_total, jnp.zeros(3)])

    return sdot



def relative_dynamics(s, t, u, mu, earth_to_sun, T,
                      pflag=[True, True, True], update_parameter=True, debug=False):
    """
    Relative Dynamics with respect to the reference orbit

    @ Inputs
    s: [9,] spacecraft state (in LVLH)  [r(3), v(3), gamma_srp, gamma_D, psi]
    t: [1,] time from start [sec]
    u: [3,] control input (in LVLH)
    mu: [1,] gravity pameter GM
    state_ref:  [4,] state of the chief (theta, theta_dot, theta_ddot, r0)
    earth_to_sun: [3,] sun direction
    state_eci_ref:  [9,] position and velocity of the chief in ECI
    T: [1,] orbital period of the reference orbit
    pflag: [3,] perturbation flag (0: SRP 1: drag, 2: thermal)
    update_parameters: [1,]  if True, propagate the dynamics propagators as well

    @ Outputs
    sdot [9,] time derivative of the states
    """
    x, y, z = s[0], s[1], s[2]
    xdot, ydot, zdot = s[3], s[4], s[5]
    gamma_srp, gamma_D, psi = s[6], s[7], s[8]
    state_eci_ref = jnp.hstack([s[9:15], s[6:9]])   # Reference Orbit state

    # First calculate the accelaration the reference orbit
    ref_dot = absolute_dynamics(state_eci_ref, t, u, mu, earth_to_sun, T,
                                pflag=[False, False, False], update_parameter=False, debug=False)

    # Calculate theta_dot, theta_ddot
    eci2rsw = rotation_eci2rsw(state_eci_ref[:3], state_eci_ref[3:6])

    # calculate ECI position of spacecraft
    eci_sc = lvlh_to_eci(state_eci_ref[:6], s[:6])  # spacecraft r and v in ECI

    # calculate perturbation
    if pflag[0]:
        a_srp = solar_radiation_pressure(gamma_srp, eci_sc[:3], earth_to_sun)
    else:
        a_srp = jnp.zeros(3)

    if pflag[1]:
        a_drag = drag_acceleration(gamma_D, eci_sc)
    else:
        a_drag = jnp.zeros(3)

    d_ECI = a_srp + a_drag  # in ECI frame
    d_rsw = eci2rsw @ d_ECI  # convert to rsw frame

    # calculate perturbated control input
    if pflag[2] and jnp.any(u):
        u_p = thermal_fluttering(psi, u, earth_to_sun, eci_sc, eci2rsw)
    else:
        u_p = u

    acc_residuals = d_rsw + u_p

    if update_parameter:
        dparam_dt = parameter_dynamics(t, T)
    else:
        dparam_dt = jnp.zeros(3)

    # Calculate theta_dot, theta_ddot
    r0 = jnp.linalg.norm(state_eci_ref[:3])
    h = jnp.linalg.norm(jnp.cross(state_eci_ref[:3], state_eci_ref[3:6]))
    theta_dot = h / r0**2
    rdot_rsw = eci2rsw @ state_eci_ref[3:6]
    theta_ddot = - 2 * rdot_rsw[0] * theta_dot / r0
    r_sc = jnp.linalg.norm(eci_sc)
    xddot =   2 * theta_dot * ydot + theta_ddot * y + theta_dot**2 * x - mu * (r0 + x)/r_sc**3 + mu/r0**2 + acc_residuals[0]
    yddot = - 2 * theta_dot * xdot - theta_ddot * x + theta_dot**2 * y - mu * y/r_sc**3 + acc_residuals[1]
    zddot = - mu * z/r_sc**3 + acc_residuals[2]

    # The parameter is updated
    sdot = jnp.hstack([xdot, ydot, zdot, xddot, yddot, zddot, dparam_dt, ref_dot[:6]])

    return sdot


def unit_sc_to_sun(r_sc, earth_to_sun):
    """
    Compute the unit vector to the sun

    @Input
    r_sc [3,] position of the spacecraft (ECI)
    earth_to_sun [3,] position vector from earth to sun

    @Ouput
    u_sun [3,] unit vector pointing sun from spacecraft
    """
    u_sun = earth_to_sun - r_sc  # (S - E) - (sc - E) = S - sc
    u_sun = u_sun / jnp.linalg.norm(u_sun)

    return u_sun


def solar_radiation_pressure(gamma_srp, r_ECI, earth_to_sun):
    """
    gamma:  C_SRP * S [m^2] / m [kg]  ->  m^2/kg
    state: [r(3), v(3), gamma_srp, gamma_D, psi]
    """
    # TODO: Add eclipse?

    c = 299792.48  # km/s
    sigma = 1380 / c   #  [W/m^2] / [km/s] = [kg m^2 s/ m^2 km s^3] = [kg / km s^2] @ 1AU

    u_sun = unit_sc_to_sun(r_ECI, earth_to_sun)

    a_srp = - sigma * gamma_srp * u_sun * 1e-6   # [kg/km s^2]x[m^2/kg] = [m^2/km s^2] -> need to multiply 1e-6

    return a_srp


def drag_acceleration(gamma_D, eci_rv):
    """
    state: [r(3), v(3), gamma_srp, gamma_D, psi]
    """
    R_EARTH = 6378
    r_s = eci_rv[:3]
    v_s = eci_rv[3:6]
    h = jnp.linalg.norm(r_s) - R_EARTH

    EarthW = jnp.array([0, 0, 7.2921158553e-5])   # Earth rotation rate [rad/s]
    v_w = jnp.cross(r_s, EarthW)   # wind velocity
    v_r = v_s - v_w    # wind relative velocity
    v_r_norm = jnp.linalg.norm(v_r)
    d = - v_r / v_r_norm  # drag direction

    # compute atomospheric density
    # Reference:
    #         https://www.spaceacademy.net.au/watch/debris/atmosmod.htm
    F_10 = 300
    Ap = 0
    T = 900 + 2.5 * (F_10 - 70) + 1.5 * Ap
    mu = 27 - 0.012 * (h - 200)
    H = T / mu
    rho = 6e-10 * jnp.exp(- (h - 175)/ H)

    # for < 180
    # a = [7.001985e-02,
    #      -4.336216e-03,
    #      -5.009831e-03,
    #      1.621827e-04,
    #      -2.471283e-06,
    #      1.904383e-08,
    #      -7.189421e-11,
    #      1.060067e-13]
    # rho = jnp.power(10, ((((((a[7] * h + a[6]) * h + a[5]) * h + a[4]) * h + a[3]) * h + a[2]) * h + a[1]) * h + a[0])

    a_drag = 1/2 * rho * gamma_D * v_r_norm**2 * d * 1e3  #  [kg/m^3]x[m^2/kg]x[km^2/s^2] = [km^2/m s^2] -> need to multiply 1000

    return a_drag


def Rx(theta):
    """
    Rotation vector around x-axis
    """
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    M = jnp.array([[1.0, 0.0, 0.0],
                  [0.0, +c, +s],
                  [0.0, -s, +c]])
    return M


def thermal_fluttering(psi, u, earth_to_sun, eci_sc, eci2rsw):
    """
    Perturbation to control due to thermal fluttering
    state: [r(3), v(3), gamma_srp, gamma_D, psi]
    """
    u_sun = unit_sc_to_sun(eci_sc[:3], earth_to_sun)

    # e_x, e_y, e_z is the local solar panel frame vector in ECI
    e_z = u_sun
    e_y = jnp.cross(u_sun, eci_sc[:2])
    e_y = e_y / jnp.linalg.norm(e_y)
    e_x = jnp.cross(e_y, e_z)

    eci2local = jnp.vstack([e_x, e_y, e_z])
    local2rsw = eci2rsw @ eci2local.transpose()  # local -> eci -> rsw

    M = local2rsw @ Rx(psi) @ local2rsw.transpose()
    u_p = M @ u

    return u_p


def parameter_dynamics(t, T):
    """
    Oscillation in the SRP, drag, and thermal fluttering parameters

    T: orbital period of the reference orbit
    """
    p = jnp.zeros(3)
    max_psi = 5 * jnp.pi / 180   # 5 deg
    C_D = 1.5   # []
    C_SRP = 1.5  # []
    A = 1.0  # m^2
    m = 50   # kg
    gamma_SRP_nominal = C_SRP * A / m
    gamma_D_nominal = C_D * A / m

    p = jnp.array([0.05 * gamma_SRP_nominal * (2*jnp.pi/T) * jnp.cos(2*jnp.pi*t/T),  # gamma SRP
                   0.05 * gamma_D_nominal * (2*jnp.pi/T) * jnp.cos(2*jnp.pi*t/T),  # gamma D
                   max_psi * (2*jnp.pi/T) * jnp.cos(2*jnp.pi*t/T)]) # psi
    return p


def propagate_theta(mu, ref_oe, tsim):
    n = np.sqrt(mu/ref_oe[0]**3)
    a = ref_oe[0]
    e = ref_oe[1]
    s0 = jnp.array([0, n / np.power((1 - e**2), 3/2) * np.power(1 + e, 2), 0, 0])

    lent = tsim.size
    s = np.zeros((lent, 4))
    s[0, :] = s0

    def ref_dynamics(s, t, mu):
        theta = s[0]
        theta_dot = s[1]
        # theta_dot = n / np.power((1 - e**2), 3/2) * np.power(1 + e * jnp.cos(theta), 2)
        theta_ddot = - 2 * n**2 * np.sin(theta) * np.power((1 + e * np.cos(theta))/ (1 - e**2), 3)
        return np.array([theta_dot, theta_ddot])

    for k in range(lent - 1):
        s[k + 1, :2] = odeint(ref_dynamics, s[k, :2], tsim[k:k + 2], (mu,))[1]
        s[k + 1, 2] = - 2 * n**2 * np.sin(s[k+1, 0]) * np.power((1 + e * np.cos(s[k+1, 0]))/ (1 - e**2), 3)

    # compute orbit radius
    r0 = a * (1 - e**2) * np.ones_like(s[:, 0]) / (1 + e * np.cos(s[:, 0]))
    s[:, 3] = r0

    return s

def compute_earth_to_sun():
    et0 = epoch_from_string('2022-01-01 12:00:00.000')   # starting time epoch

    # fix the sun position for simplicity
    earth = pk.planet.jpl_lp('earth')
    r_earth, v = earth.eph(et0)
    earth_to_sun = - jnp.array(r_earth) * 1e-3   # km
    return earth_to_sun


def eci2rsw(r_ECI,v_ECI):
    # transformation matrix (ECI to RSW)
    T = rotation_eci2rsw(r_ECI, v_ECI)

    # position anc velocity
    rRSW = T @ r_ECI
    vRSW = T @ v_ECI

    return T, rRSW, vRSW


def eci_to_lvlh(s_chief, s_deputy):
    """
    Calculate the relative position and velocity in the LVLH frame
    "Orbital Mechanics for Engineering Students", p 317
    # Coordinates
    #   - LVLH frame
    #      - R: +radial direction
    #      - S: +velocity along-track
    #      - W: +angular momentum vector (cross-track)
    """
    r_chief = s_chief[:3]
    r_deputy = s_deputy[:3]
    v_chief = s_chief[3:6]
    v_deputy = s_deputy[3:6]

    r_chief_norm = np.linalg.norm(r_chief)

    # convert deputyr to ECI -> RSW  
    M_ECI2RSW = rotation_eci2rsw(r_chief,v_chief)
    omega = jnp.cross(r_chief, v_chief) / (r_chief_norm**2)

    # show relative state in RSW (LVLH) coordinate
    r_rel_ECI = r_deputy - r_chief
    r_rel_RSW = M_ECI2RSW @ r_rel_ECI

    v_rel_ECI = (v_deputy - v_chief) - jnp.cross(omega, r_rel_ECI)  # in ECI
    v_rel_RSW = M_ECI2RSW @ v_rel_ECI

    # Convert from LVLH to orbital frame
    rv_lvlh = jnp.hstack([r_rel_RSW, v_rel_RSW])

    return rv_lvlh

def lvlh_to_eci(s_chief, lvlh_deputy):
    """
    Calculate the ECI orbit of deputy from the chief orbit and the deputy LVLH

    """
    r_chief = s_chief[:3]
    r_rel_RSW = lvlh_deputy[:3]
    v_chief = s_chief[3:6]
    v_rel_RSW = lvlh_deputy[3:6]

    # transformation matrix
    r_chief_norm = jnp.linalg.norm(r_chief)
    M_ECI2RSW = rotation_eci2rsw(r_chief, v_chief)
    omega = jnp.cross(r_chief, v_chief) / (r_chief_norm ** 2)

    r_deputy = r_chief + M_ECI2RSW.transpose() @ r_rel_RSW
    r_rel_ECI = r_deputy - r_chief
    v_deputy = v_chief + jnp.cross(omega, r_rel_ECI) + M_ECI2RSW.transpose() @ v_rel_RSW

    rv_deputy = jnp.hstack([r_deputy, v_deputy])

    return rv_deputy


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


def zero_control(t, s):
    return jnp.zeros(3)

def random_control(t, s, u_max):
    return jnp.zeros()

def sample_range(min, max):
    n = np.size(min)
    return min + np.random.sample(n) * (max - min)


# -----------------------------------------------------------
# Data generation for NN controller
# -----------------------------------------------------------
def calculate_residuals(s, t, u, mu, earth_to_sun, T, pflag, update_parameter):
    x, y, z = s[0], s[1], s[2]
    xdot, ydot, zdot = s[3], s[4], s[5]
    gamma_srp, gamma_D, psi = s[6], s[7], s[8]
    state_eci_ref = jnp.hstack([s[9:15], s[6:9]])  # Reference Orbit state

    # Calculate theta_dot, theta_ddot
    eci2rsw = rotation_eci2rsw(state_eci_ref[:3], state_eci_ref[3:6])

    # calculate ECI position of spacecraft
    eci_sc = lvlh_to_eci(state_eci_ref[:6], s[:6])  # spacecraft r and v in ECI

    # calculate perturbation
    if pflag[0]:
        a_srp = solar_radiation_pressure(gamma_srp, eci_sc[:3], earth_to_sun)
    else:
        a_srp = jnp.zeros(3)

    if pflag[1]:
        a_drag = drag_acceleration(gamma_D, eci_sc)
    else:
        a_drag = jnp.zeros(3)

    d_ECI = a_srp + a_drag  # in ECI frame
    d_rsw = eci2rsw @ d_ECI  # convert to rsw frame

    # calculate perturbated control input
    if pflag[2] and jnp.any(u):
        u_p = thermal_fluttering(psi, u, earth_to_sun, eci_sc, eci2rsw)
    else:
        u_p = u

    acc_residuals = d_rsw + u_p

    if update_parameter:
        dparam_dt = parameter_dynamics(t, T)
    else:
        dparam_dt = jnp.zeros(3)

    return acc_residuals, dparam_dt, eci_sc, eci_sc


def calculate_residuals_with_perturbations(s, t, u, mu, earth_to_sun, T):
    x, y, z = s[0], s[1], s[2]
    xdot, ydot, zdot = s[3], s[4], s[5]
    gamma_srp, gamma_D, psi = s[6], s[7], s[8]
    state_eci_ref = jnp.hstack([s[9:15], s[6:9]])  # Reference Orbit state

    # Calculate theta_dot, theta_ddot
    eci2rsw = rotation_eci2rsw(state_eci_ref[:3], state_eci_ref[3:6])

    # calculate ECI position of spacecraft
    eci_sc = lvlh_to_eci(state_eci_ref[:6], s[:6])  # spacecraft r and v in ECI

    # calculate perturbation
    a_srp = solar_radiation_pressure(gamma_srp, eci_sc[:3], earth_to_sun)
    a_drag = drag_acceleration(gamma_D, eci_sc)
    d_ECI = a_srp + a_drag  # in ECI frame
    d_rsw = eci2rsw @ d_ECI  # convert to rsw frame

    # calculate perturbated control input
    u_p = thermal_fluttering(psi, u, earth_to_sun, eci_sc, eci2rsw)

    acc_residuals = d_rsw + u_p

    dparam_dt = parameter_dynamics(t, T)

    return acc_residuals, dparam_dt


def generate_data(init_oe_range, init_rel_rv_range, u_range, param_range, n_sample):
    """
    Generate data for Neural Net training
    """

    # constants
    earth_to_sun = compute_earth_to_sun()
    R_EARTH = 6378
    mu = MU_EARTH * 1e-9
    pflag = [True, True, True]  # include all perturbations
    update_parameter = True     # update parameters

    INPUT_SIZE = 16  # state(15), time(1)
    OUT_SIZE = 6   # residual(3), parameter derivativ(3)

    datasets = np.zeros((n_sample, INPUT_SIZE+OUT_SIZE))

    calculate_residuals_jit = jax.jit(calculate_residuals_with_perturbations)

    # Sample points from the oe range
    idx = 0
    for i in range(n_sample):
        oe_ref = sample_range(init_oe_range[0,:], init_oe_range[1,:])    # sample oe for init and reference
        a = oe_ref[0]
        e = oe_ref[1]
        if (a*(1-e) - R_EARTH) < 200:
            oe_ref[0] = (R_EARTH + 200 + 500 * np.random.rand()) / (1 - oe_ref[1])
            a = oe_ref[0]

        T = 2 * jnp.pi * np.sqrt(a**3/mu)
        t = np.random.sample() * T
        r_ref, v_ref = par2ic(oe_ref)
        init_rv_rel = sample_range(init_rel_rv_range[0,:], init_rel_rv_range[1,:])
        params = sample_range(param_range[0,:], param_range[1,:])
        u = sample_range(u_range[0,:], u_range[1,:])
        s0 = np.hstack([init_rv_rel, params, np.array(r_ref), np.array(v_ref)])
        acc_residuals, dparam_dt = calculate_residuals_jit(jnp.array(s0), t, jnp.array(u), mu, earth_to_sun, T)
        #acc_residuals, dparam_dt = calculate_residuals_with_perturbations(jnp.array(s0), t, jnp.array(u), mu, earth_to_sun, T)
        datasets[idx,:] = np.hstack([s0, t, acc_residuals, dparam_dt])
        idx = idx + 1

    return datasets



# --------------------------------------------------------------
#  Plotting function
# --------------------------------------------------------------
def plot_absolute_orbits(sc, sd):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # load bluemarble with PIL
    bm = np.array(PIL.Image.open('earth_surface.jpg'))
    # it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept 
    # bm = np.array(bm.resize([d/5 for d in bm.size]))/256.

    # coordinates of the image - don't know if this is entirely accurate, but probably close
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 

    R = 6378
    x = np.outer(np.cos(lons), np.cos(lats)).T
    y = np.outer(np.sin(lons), np.cos(lats)).T
    z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    ax.plot_surface(R*x, R*y, R*z, rstride=4, cstride=4, facecolors = bm/256, alpha=0.1)

    ax.plot3D(sc[:,0], sc[:,1], sc[:,2], label='reference')
    ax.plot3D(sd[:,0], sd[:,1], sd[:,2], label='spacecraft')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.legend()


def plot_relative_orbit_from_abs(tspan, sc, sd):
    lent = tspan.size
    n = 6
    x_lvlh = np.zeros((lent,n))
    for k in range(lent):
        x_lvlh[k, :] = np.array(eci_to_lvlh(sc[k,:6], sd[k,:6]))

    plt.figure()
    plt.subplot(2,1,1)
    lab = ["x", "y", "z", "vx", "vy", "vz"]
    for i in range(3):
        plt.plot(tspan, x_lvlh[:,i],label=lab[i])
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Position  [km]')
    plt.legend()

    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(tspan, x_lvlh[:,i+3],label=lab[i+3])
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [km/s]')
    plt.legend()

    plt.savefig("relative_orbit.png")

    plt.show()

    return x_lvlh


def plot_relative_orbit(tspan, s_rel):
    plt.figure()
    plt.subplot(2, 1, 1)
    lab = ["x", "y", "z", "vx", "vy", "vz"]
    for i in range(3):
        plt.plot(tspan, s_rel[:, i], label=lab[i])
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Position  [km]')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(tspan, s_rel[:, i + 3], label=lab[i + 3])
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [km/s]')
    plt.legend()
    plt.show()