import numpy as np

def cartesian_to_polar(cart):
    """
    input format: (x_1,..., x_n, y_1,..., y_n, z_1,..., z_n)
    output format: (r_1,..., r_n, θ_1,..., θ_n, z_1,..., z_n)
    """
    n_ions = cart.shape[1] // 3
    pol = np.zeros(cart.shape)
    pol[:, 0:n_ions] = np.sqrt(cart[:, 0:n_ions]**2 + cart[:, n_ions:2*n_ions]**2) 
    pol[:, n_ions:2*n_ions] = np.arctan2(cart[:, n_ions:2*n_ions], cart[:, 0:n_ions])
    pol[:, 2*n_ions:] = cart[:, 2*n_ions:]
    return pol

def extract_tangential_velocity(r_sim, v_sim, average=True):
    n_ions = r_sim.shape[1] // 3

    r_sim_pol_r = cartesian_to_polar(r_sim)[:, :n_ions]
    v_theta = (v_sim[:,n_ions:2*n_ions]*r_sim[:,:n_ions] - v_sim[:,:n_ions]*r_sim[:,n_ions:2*n_ions]) / r_sim_pol_r
    angular_v = v_theta / r_sim_pol_r

    if average:
        return np.mean(v_theta, axis=1), np.mean(angular_v, axis=1)
    return v_theta, angular_v

def rotate_coordinates(r_sim, v_sim, dt, angular_v_avg = None):
    n_ions = r_sim.shape[1] // 3
    n_tsteps = r_sim.shape[0]

    if angular_v_avg is None:
        _, angular_v_avg = extract_tangential_velocity(r_sim, v_sim)

    r_sim_rot = r_sim.copy()
    v_sim_rot = v_sim.copy()
    tot_angle = 0

    for i in range(n_tsteps):
        tot_angle += angular_v_avg[i] * dt
        rot_mat = np.array([
            [np.cos(tot_angle), np.sin(tot_angle)],
            [-np.sin(tot_angle), np.cos(tot_angle)]
        ])
        r_sim_rot[i,:2*n_ions] = (rot_mat @ r_sim[i,:2*n_ions].reshape(2, -1)).flatten()
        v_sim_rot[i,:2*n_ions] = (rot_mat @ v_sim[i,:2*n_ions].reshape(2, -1)).flatten()

    return r_sim_rot, v_sim_rot