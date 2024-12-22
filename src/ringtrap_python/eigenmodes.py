import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from .utils import extract_tangential_velocity

def _pot_energy(positions, ensemble_properties, potentials=[], dims=3):
    """Helper function to compute the total potential energy given a list of potentials and the ion positions."""
    energy = 0
    for potential in potentials:
        energy += potential.potential(positions, ensemble_properties)
    return energy

def _jac(positions, ensemble_properties, potentials=[], dims=3):
    """Helper function to compute the gradient of the total potential energy given a list of potentials and the ion positions."""
    grad = np.zeros(dims*ensemble_properties["n"])
    for potential in potentials:
        grad += potential.jac(positions, ensemble_properties)
    return grad

def _hessian(positions, ensemble_properties, potentials=[], dims=3):
    n = ensemble_properties["n"]
    hess = np.zeros((dims*n, dims*n))
    for potential in potentials:
        hess += potential.hess(positions, ensemble_properties)
    return hess

def get_ring_eq_pos(ensemble_properties, offset=0, potentials=[], initial_radius=1e-4, method="BFGS"):
    """
    Compute the equilibrium positions of the ions in a ring trap
    offset (units: rad) is the angular offset of the first ion in the ring
    initial_radius (units: m) is the initial radius of the ion chain before minimizing the potential
    """
    n = ensemble_properties["n"]
    r_0 = np.zeros(3*n)
    for i in range(n):
        r_0[i] = initial_radius * np.cos(2*np.pi*(i/n))
        r_0[i+n] = initial_radius * np.sin(2*np.pi*(i/n))
        r_0[i+2*n] = 0
    
    if method == "BFGS":
        bfgs_tolerance = 1e-34
        opt = optimize.minimize(_pot_energy,
                                r_0,
                                args=(ensemble_properties, potentials, 3),
                                jac=_jac,
                                options={"gtol": bfgs_tolerance, "disp": False})
    return opt.x

def get_linear_eq_pos(ensemble_properties, initial_dist, potentials=[], method="BFGS"):
    """
    Compute the equilibrium positions of the ions in a 1D linear trap
    initial_dist (units: m) is the initial distance between ions before minimizing the potential.
    """
    n = ensemble_properties["n"]
    r_0 = np.zeros(n)
    for i in range(n):
        r_0[i] = i*initial_dist - (n // 2) * initial_dist
    
    if method == "BFGS":
        bfgs_tolerance = 1e-34
        opt = optimize.minimize(_pot_energy,
                                r_0,
                                args=(ensemble_properties, potentials, 1),
                                jac=_jac,
                                options={"gtol": bfgs_tolerance, "disp": False})
    return opt.x

def get_ring_eigenmodes(ensemble_properties, potentials):
    eq_positions = get_ring_eq_pos(ensemble_properties, potentials=potentials)
    hess = _hessian(eq_positions, ensemble_properties, potentials=potentials)

    eigenvalues, eigenvectors = np.linalg.eigh(hess)
    eigenfrequencies = np.sqrt(eigenvalues / ensemble_properties["mass"])
    eigenvectors = eigenvectors.T
    return eigenfrequencies, eigenvectors

def plot_ring_eigenmodes(eq_pos, eigenvectors, trap_radius, n_cols=3):
    n_ions = eigenvectors.shape[0] // 3
    n_modes = eigenvectors.shape[0]
    n_rows = (n_modes // n_cols if n_modes % n_cols == 0 else n_modes // n_cols + 1)
    eigenvectors = eigenvectors * 0.5 * trap_radius

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    for k in range(n_modes):
        ax.flat[k].add_patch(plt.Circle((0, 0), trap_radius, fill=False, color="gray"))
        ax.flat[k].set_xlim(-1.5 * trap_radius, 1.5 * trap_radius)
        ax.flat[k].set_ylim(-1.5 * trap_radius, 1.5 * trap_radius)
        ax.flat[k].set_xlabel("x (µm)")
        ax.flat[k].set_xlabel("y (µm)")
        ax.flat[k].tick_params(axis="x")
        ax.flat[k].tick_params(axis="y")
        for i in range(n_ions):
            ax.flat[k].arrow(eq_pos[i], eq_pos[n_ions+i], eigenvectors[k][i], eigenvectors[k][n_ions+i], width=0.02*trap_radius)
    
    plt.show()

def get_rotating_mode_energy(ensemble_properties, r_sim, v_sim):
    v_theta, _ = extract_tangential_velocity(r_sim, v_sim, average=False)
    return np.sum(0.5 * ensemble_properties["mass"] * v_theta * v_theta, axis=1)

def get_mode_energies(ensemble_properties, r_sim, v_sim, potentials, n_zero_freq_modes=0, dims=3):
    n = ensemble_properties["n"]
    n_tsteps = r_sim.shape[0]
    efreqs, evecs = get_ring_eigenmodes(ensemble_properties, potentials)

    mode_energies = np.zeros((n_tsteps, dims*n))
    for i in range(n_tsteps):
        for j in range(n_zero_freq_modes, dims*n):
            mode_energies[i][j] = (1/2) * ensemble_properties["mass"] * (efreqs[j] ** 2) * ( ((evecs[j].T @ r_sim[i]) ** 2) + (( (evecs[j].T @ v_sim[i]) / efreqs[j]) ** 2) ) 
    
    return mode_energies

def get_total_energy(ensemble_properties, r_sim, v_sim, potentials, dims=3):
    n_tsteps = r_sim.shape[0]
    total_energy = 0.5 * ensemble_properties["mass"] * np.sum(v_sim * v_sim, axis=1)
    for potential in potentials:
        for i in range(n_tsteps):
            total_energy[i] += potential.potential(r_sim[i], ensemble_properties)
    return total_energy
