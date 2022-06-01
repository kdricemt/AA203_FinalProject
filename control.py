# Standard imports for SCP via JAX and CVXPY
import numpy as np # CVXPY uses numpy
import cvxpy as cvx # For Convex Optimization
from cvxpy.constraints import constraint
from cvxpy.error import SolverError
import jax # For computing gradients and JIT
import jax.numpy as jnp
from tqdm import tqdm # For progress bars
from functools import partial # For helping JAX

import pykep
import dynamics as dn

# Assessment
import time
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize

"""
Common Parameters for orbital dynamics
"""
R_EARTH = 6378
MU_EARTH_KM = pykep.MU_EARTH * 1e-9
EARTH_TO_SUN_VEC = dn.compute_earth_to_sun()

"""
JAX Happy Dynamics
"""
def absolute_dynamics(s: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray):
    """
    Absolute Dynamics of the spacecraft

    @ Inputs
    s: [6,] in ECI
    t: time, unused
    u: [3,] in ECI
    """
    # print(f"s shape = {s.shape} and u shape = {u.shape}")

    # Two body dynamics
    r = s[:3]
    R = jnp.linalg.norm(r)
    a_body = -MU_EARTH_KM * r / R **3
    a_total = a_body + u * 1e-3

    sdot = jnp.hstack([s[3:6], a_total])
    return sdot


def relative_dynamics(s: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray,
                      params: jnp.ndarray, state_eci_ref_func: callable):
    """
    Relative Dynamics with respect to the reference orbit

    @ Inputs
    s: [6,] spacecraft state (in LVLH)  [r(3), v(3)]
    t: [1,] time from start [sec]
    u: [3,] control input (in LVLH)
    mu: [1,] gravity pameter GM
    state_eci_ref_func: t -> [6,] position and velocity of the chief in ECI

    @ Outputs
    sdot [9,] time derivative of the states
    """
    x, y, z = s[0], s[1], s[2]
    xdot, ydot, zdot = s[3], s[4], s[5]
    state_eci_ref = state_eci_ref_func(t).flatten()   # Reference Orbit state
    GAMMA_SRP, GAMMA_DRAG, psi = params[0], params[1], params[2]

    # First calculate the accelaration the reference orbit
    # ref_dot = absolute_dynamics(state_eci_ref, t, jnp.zeros((3,)))

    # Calculate theta_dot, theta_ddot
    eci2rsw = dn.rotation_eci2rsw(state_eci_ref[:3], state_eci_ref[3:6])
    r0 = jnp.linalg.norm(state_eci_ref[:3])
    h = jnp.linalg.norm(jnp.cross(state_eci_ref[:3], state_eci_ref[3:6]))
    theta_dot = h / r0**2
    rdot_rsw = eci2rsw @ state_eci_ref[3:6]
    theta_ddot = - 2 * rdot_rsw[0] * theta_dot / r0

    # calculate ECI position of spacecraft
    eci_sc = dn.lvlh_to_eci(state_eci_ref[:6], s[:6])   # spacecraft r and v in ECI

    # calculate perturbation
    a_srp = dn.solar_radiation_pressure(GAMMA_SRP, eci_sc[:3], EARTH_TO_SUN_VEC)
    a_drag = dn.drag_acceleration(GAMMA_DRAG, eci_sc)

    d_ECI = a_srp + a_drag   # in ECI frame
    d_rsw = eci2rsw @ d_ECI  # convert to rsw frame

    r_sc = jnp.linalg.norm(eci_sc)
    xddot =   2 * theta_dot * ydot + theta_ddot * y + theta_dot**2 * x - MU_EARTH_KM * (r0 + x)/r_sc**3 + MU_EARTH_KM/r0**2 + d_rsw[0] + u[0] * 1e-3
    yddot = - 2 * theta_dot * xdot - theta_ddot * x + theta_dot**2 * y - MU_EARTH_KM * y/r_sc**3 + d_rsw[1] + u[1] * 1e-3
    zddot = - MU_EARTH_KM * z/r_sc**3 + d_rsw[2] + u[2] * 1e-3

    # The parameter is updated
    sdot = jnp.hstack([xdot, ydot, zdot, xddot, yddot, zddot])

    return sdot

"""
SIMPLE Control Strategy
"""
def zero_control(t):
  return jnp.zeros((3, t.shape[0]))

def const_control(t, const_val):
  return jnp.tile(const_val[:, jnp.newaxis], t.shape[0])

def randn_control(t, mu, std):
  return jnp.array(mu + std * np.random.randn(3, t.shape[0]))

def max_control(t, ub):
    return ub * jnp.ones((3, t.shape[0]))

def rand_box_control(t, lb, ub):
  bound_diff = ub - lb
  return jnp.array((ub - lb) * np.random.random((3, t.shape[0])) + lb)

"""
Interporation Function
"""
def interp_many(t_vals, s_vals, t_query):
  s_query = [jnp.interp(t_query, t_vals, s_vals[i, :]) for i in range(s_vals.shape[0])]
  return jnp.array(s_query)


def interp_fixed(t_vals, s_vals):

  def interp_wrapped(t_query):
    return interp_many(t_vals, s_vals, t_query)

  return interp_wrapped


"""
JAX linearization
"""
@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def linearize(fd: callable,
              s: jnp.ndarray,
              t: jnp.ndarray,
              u: jnp.ndarray):
    """
    Linearize the a discrete time dynamics function `fd(s,u)` around `(s,u)`.

    :param fd: The discrete time dynamics
    :param s: The state (can be any dimension as long as 1D)
    :param t: The time (1D)
    :param u: The control (can be any dimension as long as 1D)

    :return: The matrices A, B, and c
    """
    # Get the state and control dimensions
    n, m = s.size, u.size

    f_point = fd(s, t, u)

    # State Linearization
    A = jax.jacrev(fd, 0)(s, t, u)
    assert A.shape == (n, n), f"A shape is {A.shape}. Expected {(n, n)}."

    # Control Linearization
    B = jax.jacrev(fd, 2)(s, t, u)
    assert B.shape == (n, m), f"B shape is {B.shape}. Expected {(n, m)}."

    # The Taylor Series offset
    c = f_point - A @ s - B @ u
    assert c.shape == (n,), f"c shape is {c.shape}. Expected {(n,)}."

    return A, B, c


def discretize(f, dt, **kwargs):
    """
    Discretize continuous-time dynamics `f` via Runge-Kutta integration.

    :param f: A function of the form f(s, u) = s dot
    (i.e., f calculates the state's time-derivative at state s and control u)
    :param dt: The time between states, ideally small.
    """

    def integrator(s, t, u, dt=dt):
        """
        Output dicretized function of the form
        s_k+1 = integrator(s_k, u_k)

        dt can be changed to make it more asynchronous
        """
        # This is h k1 on Wikipedia, similarly below.
        k1 = dt * f(s, t, u, **kwargs)
        k2 = dt * f(s + k1 / 2, t + dt / 2, u, **kwargs)
        k3 = dt * f(s + k2 / 2, t + dt / 2, u, **kwargs)
        k4 = dt * f(s + k3, t + dt, u, **kwargs)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator

def discretize_Euler(f, dt, **kwargs):
    """
    Discretize continuous-time dynamics `f` via Euler integration.
    """

    def integrator(s, t, u, dt=dt):
        """
        Output dicretized function of the form
        s_k+1 = integrator(s_k, u_k)

        dt can be changed to make it more asynchronous
        """
        ds = dt * f(s, t, u, **kwargs)

        return s + ds

    return integrator



def relative_dynamics_neural_net(s, t, u,
                                 params,
                                 state_eci_ref_func,
                                 model,
                                 min_max_in,
                                 min_max_out):

    sdot = relative_dynamics(s, t, u, params=np.zeros(3),
                             state_eci_ref_func=state_eci_ref_func)

    # construct inputs for NN
    state_eci_ref = state_eci_ref_func(t).flatten()  # Reference Orbit state
    nn_input = np.hstack([s, params[:2], np.array(state_eci_ref)])
    nn_in_normalized = normalize_input(nn_input, min_max_in)

    # predicte residuals using neural net
    acc_nn_normalized = model.predict(nn_in_normalized, verbose=0)[0]
    acc_nn = undo_normalize_nn_output(acc_nn_normalized, min_max_out)

    acc_nn = acc_nn[:3] + acc_nn[3:]  # srp + drag

    sdot_total = sdot + np.hstack([np.zeros(3), acc_nn])

    return sdot_total

def normalize_input(state_in, min_max):
    X_data_normalized = (state_in - min_max[0,:])/(min_max[1,:] - min_max[0,:])
    reshaped_X = np.reshape(X_data_normalized, (1, -1))
    return reshaped_X

def undo_normalize_nn_output(state_out, min_max):
    X_data_unnormalized = state_out * (min_max[1,:] - min_max[0,:]) + min_max[0,:]
    return X_data_unnormalized


def propagate_absolute_from_control(s0, u_func, t_span, fd_absolute):
    s_propagated = np.zeros((s0.shape[0], t_span.shape[0] + 1))
    s_propagated[:, 0] = s0

    u_vec_mat = u_func(t_span)

    for (t_ind, t) in enumerate(t_span):
        s_propagated[:, t_ind + 1] = fd_absolute(s_propagated[:, t_ind], t, u_vec_mat[:, t_ind])

    return s_propagated


def propagate_relative_from_control(s0, u_vec, t_span, fd_relative_4T):
    """
    fd_relative_4T = jax.jit(discretize(relative_dynamics, dt, state_eci_ref_func=interp_abs_chief_jit))
    """
    s_propagated = np.zeros((s0.shape[0], t_span.shape[0] + 1))
    s_propagated[:, 0] = s0

    # print(f"Original shape is {u_vec.shape}")
    u_vec_mat = u_vec.reshape((-1, 3)).transpose()   # 3 x T
    # print(f"New shape is {u_vec_mat.shape}")

    for (t_ind, t) in enumerate(t_span):
        # print(f"u curr (shape = {u_vec_mat[:, t_ind].shape}) = {u_vec_mat[:, t_ind]}")
        s_propagated[:, t_ind + 1] = fd_relative_4T(s_propagated[:, t_ind], t, u_vec_mat[:, t_ind])

    return s_propagated

"""
Scipy Minimize
"""
def optimize_with_scipy(N_horizon, control_interval, dt, u_min, u_max, s0_rel_sat, fd_relative_4T,
                        u_guess=None, use_MBH=False, maxiter=50, stage_w=1e8, terminal_w=1e10):
    tspan_optimize = np.arange(0, N_horizon * control_interval * dt, dt)
    t_control = np.arange(0, N_horizon * control_interval * dt, control_interval * dt)
    if u_guess is not None:
        u_example = u_guess
    else:
        u_example = zero_control(t_control)  # 3 x T
        u_example = u_example.flatten()
    lent = t_control.size

    # to be minimized from control -> control u
    def to_be_minimized_from_control(u_ctrl):
        u_mat = u_ctrl.reshape((-1, 3))  # T x 3
        norm_u_vec = np.linalg.norm(u_mat, ord=1, axis=1)
        u_cost = np.sum(norm_u_vec) * control_interval
        u_mat_rep = np.repeat(u_mat, control_interval, axis=0)   # (intervalxT) x 3
        u_ctrl_rep = u_mat_rep.flatten()
        s_propagated = propagate_relative_from_control(s0_rel_sat, u_ctrl_rep, tspan_optimize, fd_relative_4T) * 1000
        s_stage = stage_w * np.sum(s_propagated[:6, :-1].T @ s_propagated[:6, :-1])
        s_terminal = terminal_w * s_propagated[:6, -1].T @ s_propagated[:6, -1]

        return u_cost + s_stage + s_terminal

    def u_constraint(u_ctrl):
        u_mat = u_ctrl.reshape((-1, 3))  # T x 3
        norm_u_vec = np.linalg.norm(u_mat, ord=1, axis=1)
        return norm_u_vec.flatten()

    nlc = scipy.optimize.NonlinearConstraint(u_constraint, np.zeros(lent), u_max * np.ones(lent))

    print(f"Lower bound = {u_min}, Upper bound = {u_max}")
    print(f"T start = {tspan_optimize[0]}, T end = {tspan_optimize[-1]}")
    print(f"U example shape = {u_example.shape}")
    # print(f"u_example = {u_example}")
    print(f"initial cost is {to_be_minimized_from_control(u_example)}")
    print(f"Zero u cost is  {to_be_minimized_from_control(np.zeros_like(u_example))}")
    print(
        f"Min u cost is   {to_be_minimized_from_control(u_min * np.ones_like(u_example))}")
    print(
        f"Max u cost is   {to_be_minimized_from_control(u_max * np.ones_like(u_example))}")

    t_start = time.time()

    if use_MBH:
        result = scipy.optimize.basinhopping(to_be_minimized_from_control, u_example, niter=maxiter)
    else:
        result = scipy.optimize.minimize(to_be_minimized_from_control, u_example,
                                         # method="Nelder-Mead",
                                         options={'disp': True, 'maxiter': maxiter,
                                                  # 'gtol': 2e-10, 'eps': 2e-10, 'maxfun': np.inf
                                                  },
                                         bounds=scipy.optimize.Bounds(u_min, u_max, keep_feasible=False),
                                         # constraints=(nlc,)
                                         )

    t_end = time.time()
    print()
    print()
    # print(result)

    print("\nResult\n")
    print(f"Wall time = {t_end - t_start} seconds")
    print(f"final cost: {result.fun}")
    print(f"message: {result.message}")
    print(f"result iters: nit = {result.nit} with nfev = {result.nfev}")
    print(f"status: {result.status}")

    u_optimal = result.x

    # repmat u
    u_mat = u_optimal.reshape((-1, 3))  # T x 3
    u_mat_rep = np.repeat(u_mat, control_interval, axis=0)  # (intervalxT) x 3
    u_optimal = u_mat_rep.flatten()

    u_mat = u_example.reshape((-1, 3))
    u_mat_rep = np.repeat(u_mat, control_interval, axis=0)  # (intervalxT) x 3
    u_initial = u_mat_rep.flatten()

    s_optimal = propagate_relative_from_control(s0_rel_sat, u_optimal, tspan_optimize, fd_relative_4T)

    return s_optimal, u_optimal, tspan_optimize, u_initial


def shooting_with_scipy(N_horizon, control_interval, dt, u_min, u_max, s0_rel_sat, fd_relative_4T,
                        u_guess=None, maxiter=50):

    tspan_optimize = np.arange(0, N_horizon * control_interval * dt, dt)
    t_control = np.arange(0, N_horizon * control_interval * dt, control_interval * dt)
    if u_guess is not None:
        u_example = u_guess
    else:
        u_example = zero_control(t_control)  # 3 x T
        u_example = u_example.flatten()
    lent = t_control.size

    # to be minimized from control -> control u
    def to_be_minimized_from_control(u_ctrl):
        u_mat = u_ctrl.reshape((-1, 3))  # T x 3
        norm_u_vec = np.linalg.norm(u_mat, ord=1, axis=1)
        u_cost = np.sum(norm_u_vec) * control_interval

        return u_cost

    def jacob_obj(u_ctrl):
        return np.sign(u_ctrl) * u_ctrl

    # constraint -> set position and velocity to 0
    def constraint_control(u_ctrl):
        u_mat = u_ctrl.reshape((-1, 3))  # T x 3
        norm_u_vec = np.linalg.norm(u_mat, ord=1, axis=1)
        u_cost = np.sum(norm_u_vec) * control_interval
        u_mat_rep = np.repeat(u_mat, control_interval, axis=0)   # (intervalxT) x 3
        u_ctrl_rep = u_mat_rep.flatten()
        s_propagated = propagate_relative_from_control(s0_rel_sat, u_ctrl_rep, tspan_optimize, fd_relative_4T) * 1000
        return np.linalg.norm(s_propagated[:6, -1] * 1000, np.inf)

    # def jacob_constraint(u_ctrl):
    #     s_bar = np.zeros((N_horizon, n))
    #     s_bar[0] = s0_rel_sat
    #     for k in range(N_horizon-1):
    #         s_bar[k + 1] = fd_relative_4T(s_bar[k], tspan_optimize[k], u_ctrl[3*k:3*k+3])
    #
    #     A, B, c = linearize(fd_relative_4T, s_bar, tspan_optimize, u_ctrl.reshape((-1, 3)))
    #     A, B, c = np.array(A), np.array(B), np.array(c)   # B = T x 6 x 3
    #     B = B.transpose((1, 0, 2))  # T x 6 x 3 -> 6 x T x 3
    #     B = B.flatten().reshape((n, -1))
    #
    #     return B

    print(f"Lower bound = {u_min}, Upper bound = {u_max}")
    print(f"T start = {tspan_optimize[0]}, T end = {tspan_optimize[-1]}")
    print(f"U example shape = {u_example.shape}")
    # print(f"u_example = {u_example}")

    print(f"initial cost is {to_be_minimized_from_control(u_example)}")
    print(f"Zero u cost is  {to_be_minimized_from_control(np.zeros_like(u_example))}")
    print(
        f"Min u cost is   {to_be_minimized_from_control(u_min * np.ones_like(u_example))}")
    print(
        f"Max u cost is   {to_be_minimized_from_control(u_max * np.ones_like(u_example))}")

    print(f"initial cv is {constraint_control(u_example)}")
    print(f"Zero u cv is  {constraint_control(np.zeros_like(u_example))}")
    print(
        f"Min u cv is   {constraint_control(u_min * np.ones_like(u_example))}")
    print(
        f"Max u cv is   {constraint_control(u_max * np.ones_like(u_example))}")

    # test jacob constraint

    t_start = time.time()

    tol = 1e-3
    print("Start Optimization")
    nlc = scipy.optimize.NonlinearConstraint(constraint_control, 0.0, 0.0)

    result = scipy.optimize.minimize(to_be_minimized_from_control, u_example,
                                     jac=jacob_obj,
                                     options={'disp': True, 'maxiter': maxiter},
                                     bounds=scipy.optimize.Bounds(u_min, u_max, keep_feasible=True),
                                     constraints=(nlc,) )

    t_end = time.time()

    print()
    print()
    # print(result)

    print("\nResult\n")
    print(f"Wall time = {t_end - t_start} seconds")
    print(f"final cost: {result.fun}")
    print(f"message: {result.message}")
    print(f"result iters: nit = {result.nit} with nfev = {result.nfev}")
    print(f"status: {result.status}")

    u_optimal = result.x

    # repmat u
    u_mat = u_optimal.reshape((-1, 3))  # T x 3
    u_mat_rep = np.repeat(u_mat, control_interval, axis=0)  # (intervalxT) x 3
    u_optimal = u_mat_rep.flatten()

    u_mat = u_example.reshape((-1, 3))
    u_mat_rep = np.repeat(u_mat, control_interval, axis=0)  # (intervalxT) x 3
    u_initial = u_mat_rep.flatten()

    cost = to_be_minimized_from_control(result.x)
    constraint_v = constraint_control(result.x)

    s_optimal = propagate_relative_from_control(s0_rel_sat, u_optimal, tspan_optimize, fd_relative_4T)

    return s_optimal, u_optimal, tspan_optimize, u_initial, cost, constraint_v


"""
Sequential Convex Programming
"""

def solve_satellite_scp(fd: callable, t_span: np.ndarray,
                        P: np.ndarray, Q: np.ndarray,
                        N: int, s0: np.ndarray, u_max: float, rho: float,
                        tol: float, max_iters: int, u_dim: int = 3,
                        s_bar_warm=None, u_bar_warm=None):
    """
    Solve the Satellite Trajectory Tracking problem via Sequential Convex
    Programming. Note that this code is largely based on the code from AA203
    HW 3 problem 2.

    The state is 15 dimensional with
    (position in R^3, velocity in R^3, parameters in R^3,
    reference orbit position in R^3, reference orbit velocity in R^3)

    The control is 3 dimensional

    :param fd: The discretized dynamics function (completely implemented in JAX)
    :param t_span: The time span as an array (e.g., linspace)
    :param P: The terminal state cost matrix (6 by 6)
    :param Q: The stage state cost matrix (6 by 6)
    :param N: The horizon (number of time steps)
    :param s0: The initial state (15,)
    :param u_max: The maximum allowable control
    :param rho: The trust region
    :param max_iters: The maximum number of iterations
    :param u_dim: The control dimension (3,)

    :return: The optimized state and control sequence
    """

    n = s0.shape[0]  # state dimension
    m = u_dim  # control dimension

    # Initialize nominal trajectories
    if u_bar_warm is None:
        u_bar = np.zeros((N, m))
    else:
        # print("Warm start on the control")
        u_bar = u_bar_warm
        assert u_bar.shape == (N, m)

    if s_bar_warm is None:
        s_bar = np.zeros((N + 1, n))
        s_bar[0] = s0

        for k in range(N):
            s_bar[k + 1] = fd(s_bar[k], t_span[k], u_bar[k])
    else:
        s_bar = s_bar_warm

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    obj_prev = np.inf

    u_constraint_active = False

    # prog_bar = tqdm(range(max_iters), desc='SCP Iterations', position=1, leave=True)
    # prog_bar = tqdm(range(max_iters), desc='SCP Iterations')

    for i in range(max_iters):
        # Get the optimized values after one sequential convex iteration
        s, u, obj = satellite_scp_iteration(fd, t_span, P, Q, N, u_bar, s_bar, s0,
                                            u_max, rho, u_dim, u_constraint_active)
        # How much the objective (i.e., cost) changed
        diff_obj = np.abs(obj - obj_prev)
        # prog_bar.set_postfix({'objective change': '{:.5f}'.format(diff_obj)})
        print("iteration{:1d} objective change: {:.5f}".format(i, diff_obj))

        if not u_constraint_active:
            u_constraint_active = True
            np.copyto(s_bar, s)
            np.copyto(u_bar, u)
        elif diff_obj < tol:
            converged = True

            conv_message = f"SCP converged after {i} iterations."
            print(conv_message)
            # tqdm.tqdm.write(conv_message)
            # prog_bar.set_postfix({'SCP converged in:': f"{i} iterations"})
            break
        else:
            obj_prev = obj
            np.copyto(s_bar, s)
            np.copyto(u_bar, u)

    if not converged:
        raise RuntimeError('SCP did not converge!')

    # prog_bar.close()

    # Output the optimized state and control sequence
    return s, u


def satellite_scp_iteration(fd: callable,
                            t_span: jnp.ndarray,
                            P: np.ndarray, Q: np.ndarray,
                            N: int, u_bar: np.ndarray,
                            s_bar: np.ndarray, s0: np.ndarray,
                            u_max: float, rho: float, u_dim: int,
                            u_constraint_active: bool = True,
                            max_osqp_iters: int = int(1e6),
                            eps_abs_osqp: float = 1e-7,
                            eps_rel_osqp: float = 1e-7):
    """Solve a single SCP sub-problem for the cart-pole swing-up problem."""
    """
    Solve a single Sequential Convex Programming subproblem of the Satellite 
    Trajectory Tracking problem. Note that this code is largely based on the 
    code from AA203 HW 3 problem 2.

    The state is 15 dimensional with 
    (position in R^3, velocity in R^3, parameters in R^3, 
    reference orbit position in R^3, reference orbit velocity in R^3)

    The control is 3 dimensional

    :param fd: The discretized dynamics function (completely implemented in JAX)
    :param P: The terminal state cost matrix (6 by 6)
    :param Q: The stage state cost matrix (6 by 6)
    :param N: The horizon (number of time steps)
    :param s0: The initial state
    :param u_max: The maximum allowable control
    :param rho: The trust region
    :param max_iters: The maximum number of iterations
    :param u_dim: The control dimension
    """
    # Get the linearization of the dynamics with respect to state and control.
    # Convert to numpy array for CVX

    A, B, c = linearize(fd, s_bar[:-1], t_span[:-1], u_bar)
    A, B, c = np.array(A), np.array(B), np.array(c)

    # print(f"A shape = {A.shape}, B = {B.shape}, c = {c.shape}")

    # Dimensionality
    n = s0.shape[0]
    m = u_dim

    # The optimization variables
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # print(f"s_cvx shape {(N + 1, n)} and u_cvx shape {(N, m)}")

    # First, write all the constraints
    constraints = []

    # Initial point
    constraints += [s_cvx[0] == s0]

    # Dynamics
    constraints += [s_cvx[i + 1] == A[i] @ s_cvx[i] + B[i] @ u_cvx[i] + c[i] for i in range(N)]

    # State Trust Region constraints
    # constraints += [cvx.norm(s_cvx - s_bar, "inf") <= rho]

    # Control Trust Region constraints
    # constraints += [cvx.norm(u_cvx - u_bar, "inf") <= rho]

    # Control Constraints
    if u_constraint_active:
        # print(f"u_max is {u_max}")
        constraints += [cvx.norm(u_cvx[i], "inf") <= u_max for i in range(N)]

    # Second, write the objective
    objective_control = cvx.norm(u_cvx, 1)
    objective_stage_cost = cvx.sum([cvx.quad_form(s_cvx[i, :6]*1000, Q) for i in range(N)])
    objective_terminal_state = cvx.quad_form(s_cvx[-1, :6]*1000, P)

    objective = objective_control + objective_stage_cost + objective_terminal_state

    # Form the Convex Optimization problem
    prob = cvx.Problem(cvx.Minimize(objective), constraints)

    try:
        prob.solve(verbose=False, max_iter=max_osqp_iters, eps_abs=eps_abs_osqp, eps_rel=eps_rel_osqp)

        if prob.status != 'optimal':
            if prob.status == 'optimal_inaccurate':
                print("Optimal Inaccurate iteration")

    except SolverError as e:
        print(f'ITERATION FAILED!! message is:\n\n {e}')
        # print(prob)
        # raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
        eps_abs_osqp = 1e-3
        eps_rel_osqp = 1e-3

        prob.solve(verbose=True, max_iter=max_osqp_iters, eps_abs=eps_abs_osqp, eps_rel=eps_rel_osqp)

    # Get the variable outputs
    s = s_cvx.value
    u = u_cvx.value
    obj = prob.objective.value

    if obj is None:
        obj = 1e10

    # print(f"Constraint u -> {np.linalg.norm(u, np.inf)}")

    return s, u, obj


"""
Run MPC
"""


def run_MPC(s0, t_terminate, N_horizon, P, Q, u_max, rho, tol, max_iters, u_dim, dt, fd_relative_4T):
    # Use this to store the MPC results
    # t_span_full is the time at each state over the whole trajectory
    t_span_full = np.arange(0.0, t_terminate + N_horizon * dt, dt)

    # Time index to t_terminate
    t_ind_last = len(t_span_full) - N_horizon
    print(
        f"t_terminate ({t_terminate}) is at index {t_ind_last} of {len(t_span_full)} with value {t_span_full[t_ind_last]}")

    # state_history stores the states that we _actually_ visit
    # might want to store the MPC predictions in the future as well
    state_history = np.zeros((s0.shape[0], t_ind_last + 1))
    state_history[:, 0] = s0

    # control_history stores the control that we _actually_ took
    control_history = np.zeros((u_dim, t_ind_last))

    # store the instanteneous state
    state_curr = s0

    # For warm-starting
    s_bar_warm = None
    u_bar_warm = None

    # prog_bar_mpc = tqdm(range(t_ind_last), desc='MPC', position=0, leave=True)
    # prog_bar_mpc = tqdm(range(t_ind_last), desc='MPC')

    for t_ind in range(t_ind_last):
        t_curr = t_span_full[t_ind]
        print(f"At time {t_curr} of {t_terminate} (index {t_ind} of {t_ind_last})")

        # For each MPC solve, extract the time array
        t_scp_extract = t_span_full[t_ind:t_ind + N_horizon + 1]
        # print(f"SCP time scale shape {t_scp_extract.shape}")

        # Use Sequential Convex Programming to calculate the MPC step
        # N_horizon_curr = t_scp_extract.size - 1
        # s_scp is (time + 1, state dim)
        # u_scp is (time, control dim)
        s_scp, u_scp = solve_satellite_scp(fd_relative_4T, t_scp_extract,
                                           P, Q, N_horizon, state_curr,
                                           u_max, rho, tol, max_iters, u_dim,
                                           s_bar_warm=s_bar_warm, u_bar_warm=u_bar_warm)

        # Extract the action we need to take now
        assert u_scp.shape[1] == u_dim
        u_step = u_scp[0]
        assert u_step.shape == (u_dim,)
        control_history[:, t_ind] = u_step

        print(
            f"\n\nu_step 2-norm is {np.linalg.norm(u_step, 2)}, 1-norm {np.linalg.norm(u_step, 1)}, inf-norm {np.linalg.norm(u_step, np.inf)}")
        print(
            f"u_max is {u_max}, constraint passed? {np.linalg.norm(u_step, np.inf) <= u_max} and overall {np.linalg.norm(u_scp, np.inf) <= u_max}")

        # Propagate to the next state (i.e., propagate without relying on CVXPY)
        state_curr = fd_relative_4T(state_curr, t_curr, u_step)
        state_history[:, t_ind + 1] = state_curr

        # Form the next warm start
        u_bar_warm = np.vstack([u_scp[1:], u_scp[-1]])
        s_bar_warm = None

        # prog_bar_mpc.update()
        # prog_bar_mpc.refresh()

    return t_span_full, state_history, control_history

"""
Scipy MPC
"""

def optimize_with_scipy_MPC(fd: callable, t_span: np.ndarray,
                        P: np.ndarray, Q: np.ndarray,
                        N: int, s0: np.ndarray, u_max: float, rho: float,
                        tol: float, max_iters: int, u_dim: int = 3,
                        s_bar_warm=None, u_bar_warm=None):

    n = s0.shape[0]  # state dimension
    m = u_dim  # control dimension
    tspan_optimize = t_span

    if u_bar_warm is None:
        u_bar = np.zeros((N, m))
        u_bar = u_bar.flatten()
    else:
        u_bar = u_bar_warm.flatten()

    # to be minimized from control -> control u
    def to_be_minimized_from_control(u_ctrl):
        u_mat = u_ctrl.reshape((-1, 3))  # T x 3
        norm_u_vec = np.linalg.norm(u_mat, ord=1, axis=1)
        u_cost = np.sum(norm_u_vec)
        s_propagated = propagate_relative_from_control(s0, u_ctrl, tspan_optimize[:-1], fd) * 1000
        s_stage = np.sum(s_propagated[:6, :-1].T @ Q @ s_propagated[:6, :-1])
        s_terminal = s_propagated[:6, -1].T @ P @ s_propagated[:6, -1]

        return u_cost + s_stage + s_terminal

    def u_constraint(u_ctrl):
        u_mat = u_ctrl.reshape((-1, 3))  # T x 3
        norm_u_vec = np.linalg.norm(u_mat, ord=1, axis=1)
        return norm_u_vec.flatten()

    nlc = scipy.optimize.NonlinearConstraint(u_constraint, np.zeros(N), u_max * np.ones(N))
    t_start = time.time()

    result = scipy.optimize.minimize(to_be_minimized_from_control, u_bar,
                                     # method="Nelder-Mead",
                                     options={'disp': True, 'maxiter': 50
                                              # 'gtol': 2e-10, 'eps': 2e-10, 'maxfun': np.inf
                                              },
                                     bounds=scipy.optimize.Bounds(-u_max, u_max, keep_feasible=False),
                                     # constraints=(nlc,)
                                     )

    t_end = time.time()

    # print("\nResult\n")
    # print(f"Wall time = {t_end - t_start} seconds")
    print(f"final cost: {result.fun}")
    # print(f"message: {result.message}")
    # print(f"result iters: nit = {result.nit} with nfev = {result.nfev}")
    # print(f"status: {result.status}")

    u_optimal = result.x.reshape((-1,3))   # T x 3
    s_optimal = propagate_relative_from_control(s0, result.x, tspan_optimize[:-1], fd)  # 6 x T

    return s_optimal, u_optimal


def run_MPC_with_scipy(s0, t_terminate, N_horizon, P, Q, u_max, rho, tol, max_iters, u_dim, dt, fd_relative_4T):
    # Use this to store the MPC results
    # t_span_full is the time at each state over the whole trajectory
    t_span_full = np.arange(0.0, t_terminate + N_horizon * dt, dt)

    # Time index to t_terminate
    t_ind_last = len(t_span_full) - N_horizon
    print(
        f"t_terminate ({t_terminate}) is at index {t_ind_last} of {len(t_span_full)} with value {t_span_full[t_ind_last]}")

    # state_history stores the states that we _actually_ visit
    # might want to store the MPC predictions in the future as well
    state_history = np.zeros((s0.shape[0], t_ind_last + 1))
    state_history[:, 0] = s0

    # control_history stores the control that we _actually_ took
    control_history = np.zeros((u_dim, t_ind_last))

    # store the instanteneous state
    state_curr = s0

    # For warm-starting
    s_bar_warm = None
    u_bar_warm = None

    # prog_bar_mpc = tqdm(range(t_ind_last), desc='MPC', position=0, leave=True)
    # prog_bar_mpc = tqdm(range(t_ind_last), desc='MPC')

    for t_ind in range(t_ind_last):
        t_curr = t_span_full[t_ind]

        # For each MPC solve, extract the time array
        t_scp_extract = t_span_full[t_ind:t_ind + N_horizon + 1]
        # print(f"SCP time scale shape {t_scp_extract.shape}")

        # Use Sequential Convex Programming to calculate the MPC step
        # N_horizon_curr = t_scp_extract.size - 1
        # s_scp is (time + 1, state dim)
        # u_scp is (time, control dim)
        s_scp, u_scp = optimize_with_scipy_MPC(fd_relative_4T, t_scp_extract,
                                               P, Q, N_horizon, state_curr,
                                               u_max, rho, tol, max_iters, u_dim,
                                               s_bar_warm=s_bar_warm, u_bar_warm=u_bar_warm)

        # Extract the action we need to take now
        assert u_scp.shape[1] == u_dim
        u_step = u_scp[0]
        assert u_step.shape == (u_dim,)
        control_history[:, t_ind] = u_step

        print(f"At time {t_curr} of {t_terminate} (index {t_ind} of {t_ind_last})")
        print("------------------------------------------------------------")
        print(
            f"u_step 2-norm is {np.linalg.norm(u_step, 2)}, 1-norm {np.linalg.norm(u_step, 1)}, inf-norm {np.linalg.norm(u_step, np.inf)}")
        print(
            f"u_max is {u_max}, constraint passed? {np.linalg.norm(u_step, np.inf) <= u_max} and overall {np.linalg.norm(u_scp, np.inf) <= u_max}")
        print()

        # Propagate to the next state (i.e., propagate without relying on CVXPY)
        state_curr = fd_relative_4T(state_curr, t_curr, u_step)
        state_history[:, t_ind + 1] = state_curr

        # Form the next warm start
        u_bar_warm = np.vstack([u_scp[1:], u_scp[-1]])
        s_bar_warm = None

        # prog_bar_mpc.update()
        # prog_bar_mpc.refresh()

    return t_span_full, state_history, control_history


def run_MPC_with_scipy_NN(s0, t_terminate, N_horizon, P, Q, u_max, rho, tol, max_iters, u_dim, dt,
                          fd_relative_4T, fd_relative_4T_NN):
    # Use this to store the MPC results
    # t_span_full is the time at each state over the whole trajectory
    t_span_full = np.arange(0.0, t_terminate + N_horizon * dt, dt)

    # Time index to t_terminate
    t_ind_last = len(t_span_full) - N_horizon
    print(
        f"t_terminate ({t_terminate}) is at index {t_ind_last} of {len(t_span_full)} with value {t_span_full[t_ind_last]}")

    # state_history stores the states that we _actually_ visit
    # might want to store the MPC predictions in the future as well
    state_history = np.zeros((s0.shape[0], t_ind_last + 1))
    state_history[:, 0] = s0

    # control_history stores the control that we _actually_ took
    control_history = np.zeros((u_dim, t_ind_last))

    # store the instanteneous state
    state_curr = s0

    # For warm-starting
    s_bar_warm = None
    u_bar_warm = None

    # prog_bar_mpc = tqdm(range(t_ind_last), desc='MPC', position=0, leave=True)
    # prog_bar_mpc = tqdm(range(t_ind_last), desc='MPC')

    for t_ind in range(t_ind_last):
        t_curr = t_span_full[t_ind]

        # For each MPC solve, extract the time array
        t_scp_extract = t_span_full[t_ind:t_ind + N_horizon + 1]
        # print(f"SCP time scale shape {t_scp_extract.shape}")

        # Use Sequential Convex Programming to calculate the MPC step
        # N_horizon_curr = t_scp_extract.size - 1
        # s_scp is (time + 1, state dim)
        # u_scp is (time, control dim)
        # for planning use NN model
        s_scp, u_scp = optimize_with_scipy_MPC(fd_relative_4T_NN, t_scp_extract,
                                               P, Q, N_horizon, state_curr,
                                               u_max, rho, tol, max_iters, u_dim,
                                               s_bar_warm=s_bar_warm, u_bar_warm=u_bar_warm)

        # Extract the action we need to take now
        assert u_scp.shape[1] == u_dim
        u_step = u_scp[0]
        assert u_step.shape == (u_dim,)
        control_history[:, t_ind] = u_step

        print(f"At time {t_curr} of {t_terminate} (index {t_ind} of {t_ind_last})")
        print("------------------------------------------------------------")
        print(
            f"u_step 2-norm is {np.linalg.norm(u_step, 2)}, 1-norm {np.linalg.norm(u_step, 1)}, inf-norm {np.linalg.norm(u_step, np.inf)}")
        print(
            f"u_max is {u_max}, constraint passed? {np.linalg.norm(u_step, np.inf) <= u_max} and overall {np.linalg.norm(u_scp, np.inf) <= u_max}")
        print()

        # Propagate to the next state (i.e., propagate without relying on CVXPY)
        # for propagation use truth model
        state_curr = fd_relative_4T(state_curr, t_curr, u_step) # + sigma * np.random.randn(6)
        state_history[:, t_ind + 1] = state_curr

        # Form the next warm start
        u_bar_warm = np.vstack([u_scp[1:], u_scp[-1]])
        s_bar_warm = None

        # prog_bar_mpc.update()
        # prog_bar_mpc.refresh()

    return t_span_full, state_history, control_history


"""
Plot functions
"""
def plot_mpc_control(t_mpc, s_mpc, u_mpc, u_max, filelabel="", is_save=False):
    print(f"Shapes: t={t_mpc.shape}, s={s_mpc.shape}, u={u_mpc.shape}, norm(u)={np.linalg.norm(u_mpc, axis=0).shape}")

    # plt.plot(tspan_optimize, np.linalg.norm(u_example.reshape(-1, 3), np.inf, axis=1), label="u initial")
    # plt.plot(t_scp[:-1], np.linalg.norm(u_optimal_scp, np.inf, axis=1), label="u SCP single")
    plt.plot(t_mpc[:u_mpc.shape[1]], np.linalg.norm(u_mpc, np.inf, axis=0), label="u MPC SCP")
    plt.xlabel("Time (s)")
    plt.ylabel("$||u||_\infty$")
    plt.axhline(u_max, c='k', linestyle='--')
    plt.legend()
    plt.tight_layout()
    if is_save:
        plt.savefig("./Fig/control_history_" + filelabel + ".png")

    plt.show()

    dn.plot_relative_orbit(t_mpc[:s_mpc.shape[1]], s_mpc.T, filelabel, is_save)


def plot_control_trajectory(s_mpc):
    # 2D Plot -----------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), dpi=200)
    print(s_mpc[0, :].shape)
    ax[0].plot(s_mpc[0, :], s_mpc[1, :])
    ax[0].scatter(s_mpc[0, 0], s_mpc[1, 0])
    ax[0].set_xlabel("x (radial)")
    ax[0].set_ylabel("y (along-track)")

    ax[1].plot(s_mpc[1, :], s_mpc[2, :])
    ax[1].scatter(s_mpc[1, 0], s_mpc[2, 0])
    ax[1].set_xlabel("y (along-track)")
    ax[1].set_ylabel("z (cross-track)")

    ax[2].plot(s_mpc[0, :], s_mpc[2, :])
    ax[2].scatter(s_mpc[0, 0], s_mpc[2, 0])
    ax[2].set_xlabel("x (radial)")
    ax[2].set_ylabel("z (cross-track)")

    # fig.suptitle("Clairevoyant MPC Trajectory\n")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)
    plt.show()

    # 3D Plot -----------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(s_mpc[0, :], s_mpc[1, :], zs=s_mpc[2, :], zdir='z', label='MPC Curve')
    plt.tight_layout()
    plt.show()


def plot_control_trajectory2(s_mpc, s_mpci, filelabel="", is_save=False):
    # 2D Plot -----------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), dpi=200)
    print(s_mpc[0, :].shape)
    ax[0].plot(s_mpc[0, :], s_mpc[1, :], label='mpc')
    ax[0].scatter(s_mpc[0, 0], s_mpc[1, 0])
    ax[0].plot(s_mpci[0, :], s_mpci[1, :], label='init')
    ax[0].scatter(s_mpci[0, 0], s_mpci[1, 0])
    ax[0].set_xlabel("x (radial)")
    ax[0].set_ylabel("y (along-track)")

    ax[1].plot(s_mpc[1, :], s_mpc[2, :])
    ax[1].scatter(s_mpc[1, 0], s_mpc[2, 0])
    ax[1].plot(s_mpci[1, :], s_mpci[2, :])
    ax[1].scatter(s_mpci[1, 0], s_mpci[2, 0])
    ax[1].set_xlabel("y (along-track)")
    ax[1].set_ylabel("z (cross-track)")

    ax[2].plot(s_mpc[0, :], s_mpc[2, :])
    ax[2].scatter(s_mpc[0, 0], s_mpc[2, 0])
    ax[2].plot(s_mpci[0, :], s_mpci[2, :])
    ax[2].scatter(s_mpci[0, 0], s_mpci[2, 0])
    ax[2].set_xlabel("x (radial)")
    ax[2].set_ylabel("z (cross-track)")

    # fig.suptitle("Clairevoyant MPC Trajectory\n")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)
    ax[0].legend()
    plt.tight_layout()

    if is_save:
        plt.savefig("./Fig/trajectory2d_" + filelabel + ".png")

    plt.show()

    # 3D Plot -----------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(s_mpc[0, :], s_mpc[1, :], zs=s_mpc[2, :], zdir='z', label='MPC Curve')
    plt.tight_layout()
    plt.show()


def plot_mpc_trajectory(s_mpc, N_total_mpc, N_horizon_mpc, filelabel="", is_save=True):
    # colored -----------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), dpi=200)

    ax[0].scatter(s_mpc[0, 0], s_mpc[1, 0])
    ax[0].set_xlabel("x (radial)")
    ax[0].set_ylabel("y (along-track)")

    ax[1].scatter(s_mpc[1, 0], s_mpc[2, 0])
    ax[1].set_xlabel("y (along-track)")
    ax[1].set_ylabel("z (cross-track)")

    ax[2].scatter(s_mpc[0, 0], s_mpc[2, 0])
    ax[2].set_xlabel("x (radial)")
    ax[2].set_ylabel("z (cross-track)")

    times = range(0, N_total_mpc + N_horizon_mpc, N_horizon_mpc)
    for t_lower, t_upper in zip(times[:-1], times[1:]):
        # print(f"{t_lower} to {t_upper}")
        ax[0].plot(s_mpc[0, t_lower:t_upper + 1], s_mpc[1, t_lower:t_upper + 1])
        ax[1].plot(s_mpc[1, t_lower:t_upper + 1], s_mpc[2, t_lower:t_upper + 1])
        ax[2].plot(s_mpc[0, t_lower:t_upper + 1], s_mpc[2, t_lower:t_upper + 1])
        # plt.pause(0.05)
        # plt.show(block=False)

    plt.tight_layout()

    plt.savefig("./Fig/trajectory_"+ filelabel + ".png")
    plt.show()

def compute_total_cost_mpc(s_mpc, u_mpc, P, Q):
    """
    Compute the total cost of the MPC trajectory
    """
    norm_u_vec = np.linalg.norm(u_mpc, ord=1, axis=0)
    u_cost = np.sum(norm_u_vec)
    s_mpc = s_mpc * 1000
    s_stage = np.sum(s_mpc[:6, :-1].T @ Q @ s_mpc[:6, :-1])
    s_terminal = s_mpc[:6, -1].T @ P @ s_mpc[:6, -1]
    total = u_cost + s_stage + s_terminal

    print("Cost of MPC Trajectory:")
    print(" stage cost (state): ", s_stage)
    print(" fuel cost         : ", u_cost)
    print(" Terminal cost     : ", s_terminal)
    print(" Total cost        : ", total)

    return total
