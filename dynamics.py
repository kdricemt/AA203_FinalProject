import numpy as np
from scipy.integrate import odeint
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pykep as pk
from pykep.core import par2ic, MU_EARTH, epoch_from_string
import PIL

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
    v = s[3:] + u
    R = np.linalg.norm(r)
    a = -mu*r/R**3
    dydt = np.hstack([v, a])
    return dydt


def relative_dynamics(s, t, u, mu, state_chief, earth_to_sun):
    """

    start_jday: julian day since J2000 of the starting simulation time
    """
    x, y, z = s[0], s[1], s[2]
    xdot, ydot, zdot = s[3], s[4], s[5]
    theta, theta_dot, theta_ddot, r0 = state_chief[0], state_chief[1], state_chief[2], state_chief[3]
    sdot = np.zeros_like(s)

    tmp = jnp.power((r0 + x)**2 + y**2 + z**2, 3/2)

    a_srp = solar_radiation_pressure(s, earth_to_sun)
    a_drag = drag_acceleration(s)
    u_p = thermal_fluttering(s, earth_to_sun)

    d = a_srp + a_drag + u_p

    xddot = 2 * theta_dot * ydot + theta_ddot * y + theta_dot**2 * x - mu * (r0 + x)/tmp + mu/r0**2 + d[0]
    yddot = - 2 * theta_dot * xdot - theta_ddot * x + theta_dot**2 * y - mu * y/tmp + d[1]
    zddot = - mu * z/tmp + d[2]

    sdot[0] = xdot
    sdot[1] = ydot
    sdot[2] = zdot
    sdot[3] = xddot
    sdot[4] = yddot
    sdot[5] = zddot

    return sdot


def solar_radiation_pressure(s, earth_to_sun):
    """
    state: [r(3), v(3), gamma_srp, gamma_D, psi]
    """
    gamma_srp = s[6]
    P_sun = 1.38    # [kW/m^2]

    r_sc = s[:3]
    u_sun = jnp.array(earth_to_sun) - r_sc  # (S - E) - (sc - E) = S - sc
    u_sun = u_sun / jnp.linalg.norm(u_sun)

    a_srp = - P_sun * gamma_srp * u_sun   # [km/s]

    return a_srp


def drag_acceleration(s):
    """
    state: [r(3), v(3), gamma_srp, gamma_D, psi]
    """
    gamma_D = s[7]
    r_s = s[:2]
    v_s = s[3:6]
    h = jnp.sqrt(r_s)

    EarthW = jnp.array([0, 0, 7.2921158553e-5])   # Earth rotation rate [rad/s]
    v_w = jnp.cross(r_s, EarthW)   # wind velocity
    v_r = v_s - v_w    # wind relative velocity
    v_r_norm = jnp.linalg.norm(v_r)
    d = - v_r / v_r_norm  # drag direction

    # compute atomospheric density
    # Reference:
    #         https://www.spaceacademy.net.au/watch/debris/atmosmod.htm
    if h >= 180:
        F_10 = 300
        Ap = 0
        T = 900 + 2.5 * (F_10 - 70) + 1.5 * Ap
        mu = 27 - 0.012 * (h - 200)
        H = T / mu
        rho = 6e-10 * jnp.exp(- (h - 175)/ H)
    else:
        a = [7.001985e-02,
             -4.336216e-03,
             -5.009831e-03,
             1.621827e-04,
             -2.471283e-06,
             1.904383e-08,
             -7.189421e-11,
             1.060067e-13]
        rho = jnp.power(10, ((((((a[7] * h + a[6]) * h + a[5]) * h + a[4]) * h + a[3]) * h + a[2]) * h + a[1]) * h + a[0])

    a_drag = 1/2 * rho * gamma_D * v_r_norm**2 * d    # [km/s]

    return a_drag




class Spacecraft:
    def __init__(self, chief_oe):
        # system parameters
        self.mu = MU_EARTH * 1e-9
        self.et0 = epoch_from_string('2022-01-01 12:00:00.000')   # starting time epoch

        # fix the sun position for simplicity
        sun = pk.planet.jpl_lp('sun')
        earth = pk.planet.jpl_lp('earth')
        r_sun, v = sun.eph(self.et0)
        r_earth, v = earth.eph(self.et0)
        self.r_sun_to_earth = (r_sun - r_earth) * 1e-3   # km

        self.chief_oe = chief_oe  # a [km], e, i, W, w, E


    def absolute_dynamics(self, s, t, u, mu):
        '''
        Compute the Acceleration for simple two body dynamics
        '''
        r = s[:3]
        v = s[3:] + u
        R = np.linalg.norm(r)
        a = -mu*r/R**3
        dydt = np.hstack([v, a])
        return dydt


    def sun_direction(self, s):
        """
        Return spacecraft to sun direction
        """
        r_sc = s[:3]
        d = jnp.array(self.r_sun_to_earth) - r_sc
        d = d / jnp.linalg.norm(d)

        return d


    def propagate_reference_state(self, tsim):
        n = np.sqrt(self.mu/self.chief_oe[0]**3)
        a = self.chief_oe[0]
        e = self.chief_oe[1]
        s0 = np.array([0, n / np.power((1 - e**2), 3/2) * np.power(1 + e, 2), 0, 0])

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
            s[k + 1, :2] = odeint(ref_dynamics, s[k, :2], tsim[k:k + 2])[1]
            s[k + 1, 2] = - 2 * n**2 * np.sin(s[k+1, 0]) * np.power((1 + e * np.cos(s[k+1, 0]))/ (1 - e**2), 3)

        # compute orbit radius
        r0 = a * (1 - e**2) * np.ones_like(s[:, 0]) / (1 + e * np.cos(s[:, 0]))
        s[:, 3] = r0

        return s








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
    """
    Calculate the relative position and velocity in the LVLH frame
    # Coordinates
    #   - LVLH frame
    #      - R: +radial direction
    #      - S: +velocity along-track
    #      - W: +angular momentum vector (cross-track)
    """

    r_target_norm  = np.linalg.norm(r_target)
    r_chaise_norm  = np.linalg.norm(r_chaise)
    v_target_norm  = np.linalg.norm(v_target)

    # convert chaiser to ECI -> RSW  
    M_ECI2RSW, dum1, dum2 = eci2rsw(r_target,v_target)
    n_target = v_target_norm/r_target_norm

    # show relative state in RSW (LVLH) coordinate
    r_rho_RSW  = M_ECI2RSW @ (r_chaise - r_target).reshape((3,1))
    v_rho_RSW = M_ECI2RSW @ (v_chaise - v_target).reshape((3,1)) - np.array([-n_target * r_rho_RSW[1], n_target*r_rho_RSW[0], 0]).reshape((3,1))

    # Convert from LVLH to orbital frame
    r_lvlh = np.array([r_rho_RSW[1], -r_rho_RSW[2], -r_rho_RSW[0]])
    v_lvlh = np.array([v_rho_RSW[1], -v_rho_RSW[2], -v_rho_RSW[0]])

    return r_lvlh, v_lvlh


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
    return np.zeros(3)


def propagate_absolute_orbit(x0, tspan, controller, mu):
    lent = tspan.size
    n = x0.size
    s = np.zeros((lent,n))
    s[0,:] = x0
    for k in range(lent-1):
        t = tspan[k]
        u = controller(t, s[k])
        s[k+1, :] = odeint(twobody_rates, s[k,:], tspan[k:k+2], (u, mu))[1]

    return s


def plot_orbits(sc, sd):
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

    plt.savefig("absolute_orbit.png")


def plot_relative_orbit_from_abs(tspan, sc, sd):
    lent = tspan.size
    n = 6
    x_lvlh = np.zeros((lent,n))
    for k in range(lent):
        r_lvlh, v_lvlh = eci_to_lvlh(sc[k,:3], sc[k, 3:], sd[k,:3], sd[k,3:])
        x_lvlh[k,:] = np.vstack([r_lvlh, v_lvlh]).flatten()

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
