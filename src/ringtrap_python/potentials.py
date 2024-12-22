import numpy as np
from ..ringtrap_cython import potentials as cpotentials

"""
All positions arrays are specified in the following format
[x_1,..., x_n, y_1,..., y_n, z_1,..., z_n], where (x_k, y_k, z_k) are the cartesian coordinates of the kth ion in the chain.
"""

# Global physical constants
eps0 = 8.854187817e-12
k = 1 / (4 * np.pi * eps0)

class RingTrapPotential:
    """Ring trap trapping potential. Working in the pseudopotential approximation."""

    def __init__(self, trap_radius, wr, wz):
        """
            trap_radius (units: m) is the distance from the center of the trap to the desired potential minimum around the perimeter of the trapping region.
            wr (units: rad/s) is the angular frequency of the harmonic pseudopotential in the radial direction
            wz (units: rad/s) is the angular frequency of the harmonic pseudopotential in the direction perpendicular to the trap
        """
        self.trap_radius = trap_radius
        self.wr = wr
        self.wz = wz

    def potential(self, positions, ensemble_properties, language="python"):
        """
        Computes the trapping potential given the coordinates of all ions in the chain.
        """
        if language == "python":
            n = ensemble_properties["n"]
            r_coords = np.sqrt(positions[:n]**2 + positions[n:2*n]**2) # Get distance from trap center

            # Energy from harmonic pseudopotential
            energy = np.sum(0.5 * ensemble_properties["mass"] * ( (self.wr ** 2) * ((r_coords - self.trap_radius) ** 2) + (self.wz **2) * (positions[2*n:3*n] ** 2) ))
            return energy
        else:
            return cpotentials.ring_trap_potential(ensemble_properties["mass"], 
                                                   self.trap_radius,
                                                   self.wr, 
                                                   self.wz,
                                                   positions)

    def jac(self, positions, ensemble_properties, language="python"):
        """
        Computes the gradient vector of the trapping potential given the coordinates of all ions in the chain.
        The returned gradient array follows the same coordinate ordering as the positions array.
        """
        if language == "python":
            n = ensemble_properties["n"]
            grad = np.zeros(3*n)
            x = positions[:n]
            y = positions[n:2*n]
            z = positions[2*n:3*n]
            r_coords = np.sqrt(x**2 + y**2)
            grad[:n] = ((self.wr)**2) * (r_coords - self.trap_radius) * (x / r_coords)
            grad[n:2*n] = ((self.wr)**2) * (r_coords - self.trap_radius) * (y / r_coords)
            grad[2*n:3*n] = ((self.wz) ** 2) * z
            return ensemble_properties["mass"] * grad
        else:
            return np.array(cpotentials.ring_trap_jac(ensemble_properties["mass"],
                                                      self.trap_radius,
                                                      self.wr, 
                                                      self.wz,
                                                      positions))
    
    def hess(self, positions, ensemble_properties, language="python"):
        if language == "python":
            n = ensemble_properties["n"]
            x = positions[:n]
            y = positions[n:2*n]
            # z = positions[2*n:3*n]

            hessian = np.zeros((3*n, 3*n))
            for i in range(n):
                hessian[i][i] += (self.wr ** 2) * (1 - (self.trap_radius * (y[i] ** 2)) / ((x[i] ** 2 + y[i] ** 2) ** (3/2)))
                hessian[n+i][n+i] += (self.wr ** 2) * (1 - (self.trap_radius * (x[i] ** 2)) / ((x[i] ** 2 + y[i] ** 2) ** (3/2)))
                hessian[2*n+i][2*n+i] += (self.wz ** 2)
                hessian[i][n+i] += ((self.wr ** 2) * self.trap_radius * x[i] * y[i]) / ((x[i] ** 2 + y[i] ** 2) ** (3/2))
                hessian[n*i][i] += ((self.wr ** 2) * self.trap_radius * x[i] * y[i]) / ((x[i] ** 2 + y[i] ** 2) ** (3/2))

            return ensemble_properties["mass"] * hessian
        else:
            return None

    def force(self, positions, ensemble_properties, language="python"):
        """Computes the force on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language)

    def acceleration(self, positions, ensemble_properties, language="python"):
        """Computes the accleration on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class RingTrapHarmonicPotential:
    """Second-order approximation of ring trap trapping potential in cartesian coordiantes.
       Expansion is around the ring trap potential minimum, so first-order derivatives vanish."""

    def __init__(self, trap_radius, wr, wz, r0):
        """
            trap_radius (units: m) is the distance from the center of the trap to the desired potential minimum around the perimeter of the trapping region.
            wr (units: rad/s) is the angular frequency of the harmonic pseudopotential in the radial direction
            wz (units: rad/s) is the angular frequency of the harmonic pseudopotential in the direction perpendicular to the trap
            r0 (units: m) is the equlibrium coordinates around which to expand the potential
        """
        self.trap_radius = trap_radius
        self.wr = wr
        self.wz = wz
        self.r0 = r0

        n = r0.size // 3
        x = r0[:n]
        y = r0[n:2*n]

        # Compute matrix of second-order partial derivatives
        self.hess = np.zeros((3*n, 3*n))
        for i in range(n):
            self.hess[i][i] = wr * wr * (1 - ( (trap_radius * y[i] * y[i]) / ( (x[i]*x[i] + y[i]*y[i]) ** (3/2) ) ))
            self.hess[n+i][n+i] = wr * wr * (1 - ( (trap_radius * x[i] * x[i]) / ( (x[i]*x[i] + y[i]*y[i]) ** (3/2) ) ))
            self.hess[i][n+i] = (wr * wr * x[i] * y[i]) / ( (x[i]*x[i] + y[i]*y[i]) ** (3/2) )
            self.hess[n+i][i] = (wr * wr * x[i] * y[i]) / ( (x[i]*x[i] + y[i]*y[i]) ** (3/2) )
            self.hess[2*n+i][2*n+i] = wz*wz
    
    def potential(self, positions, ensemble_properties, language="python"):
        """
        Computes the harmonic approximation of the trapping potential given the coordinates of all ions in the chain.
        """
        if language=="python":
            n = ensemble_properties["n"]
            mass = ensemble_properties["mass"]
            energy = 0
            for i in range(n):
                energy += 0.5 * mass * self.hess[i][i] * (positions[i] - self.r0[i]) * (positions[i] - self.r0[i])
                energy += 0.5 * mass * self.hess[n+i][n+i] * (positions[n+i] - self.r0[n+i]) * (positions[n+i] - self.r0[n+i])
                energy += 0.5 * mass * self.hess[2*n+i][2*n+i] * (positions[2*n+i] - self.r0[2*n+i]) * (positions[2*n+i] - self.r0[2*n+i])
                energy += mass * self.hess[i][n+i] * (positions[i] - self.r0[i]) * (positions[n+i] - self.r0[n+i])
            return energy
        return None
    
    def jac(self, positions, ensemble_properties, language="python"):
        """
        Computes the gradient vector of the harmonic approximation of the trapping potential given the coordinates of all ions in the chain.
        The returned gradient array follows the same coordinate ordering as the positions array.
        """
        if language=="python":
            n = ensemble_properties["n"]
            mass = ensemble_properties["mass"]
            grad = np.zeros(3*n)
            for i in range(n):
                grad[i] = mass * self.hess[i][i] * (positions[i] - self.r0[i]) + mass * self.hess[i][n+i] * (positions[n+i] - self.r0[n+i])
                grad[n+i] = mass * self.hess[n+i][i] * (positions[i] - self.r0[i]) + mass * self.hess[n+i][n+i] * (positions[n+i] - self.r0[n+i])
                grad[2*n+i] = mass * self.hess[2*n+i][2*n+i] * (positions[2*n+i] - self.r0[2*n+i])
            return grad
        return None
    
    def force(self, positions, ensemble_properties, language="python"):
        """Computes the force on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language)

    def acceleration(self, positions, ensemble_properties, language="python"):
        """Computes the accleration on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class MutualCoulombPotential:
    """
    Potential due to Coulomb interations among all ions in the chain.
    """

    def __init__(self, cutoff_distance):
        """
        cutoff_distance (units: m) is the distance beyond which ion-ion Coulomb interactions are neglected for computational efficiency. 
        """
        self.cutoff_distance = cutoff_distance # not implemented in potentials yet

    def potential(self, positions, ensemble_properties, language="python"):
        """
        Computes the potential due to the ions' Coulomb interactions given the coordinates of all ions in the chain.
        """
        if language == "python":
            n = ensemble_properties["n"]
            energy = 0
            for i in range(n):
                for j in range(i+1, n):
                    ri = positions[i::n]
                    rj = positions[j::n]
                    energy += ((ensemble_properties["charge"] ** 2) / (4*np.pi*eps0)) * (1 / np.linalg.norm(ri - rj))
            return energy
        else:
            return cpotentials.mutual_coulomb_potential(ensemble_properties["charge"], positions)

    def jac(self, positions, ensemble_properties, language="python"):
        """
        Computes the gradient vector of the Coulomb potential given the coordinates of all ions in the chain.
        The returned gradient array follows the same coordinate ordering as the positions array.
        """
        if language == "python":
            n = ensemble_properties["n"]
            grad = np.zeros(3*n)
            for i in range(n):
                for j in range(i+1,n):
                    ri = positions[i::n]
                    rj = positions[j::n]
                    coulomb = ((ensemble_properties["charge"] ** 2) / (4*np.pi*eps0)) * (1 / (np.linalg.norm(ri - rj) ** 2))
                    dr = (ri - rj) / np.linalg.norm(ri - rj)
                    grad[i::n] -= coulomb * dr 
                    grad[j::n] += coulomb * dr
            return grad
        else:
            return np.array(cpotentials.mutual_coulomb_jac(ensemble_properties["charge"], positions))

    def hess(self, positions, ensemble_properties, language="python"):
        if language == "python":
            n = ensemble_properties["n"]
            x = positions[:n]
            y = positions[n:2*n]
            z = positions[2*n:3*n]

            dist2 = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist2[i,j] = ((x[i] - x[j]) ** 2) + ((y[i] - y[j]) ** 2) + ((z[i] - z[j]) ** 2)

            hessian = np.zeros((3*n, 3*n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        for k in range(n):
                            if k != i:
                                hessian[i][i] -= ((dist2[i,k] - 3*((x[i] - x[k]) ** 2)) / (dist2[i,k] ** (5/2)))
                                hessian[n+i][n+i] -= ((dist2[i,k] - 3*((y[i] - y[k]) ** 2)) / (dist2[i,k] ** (5/2)))
                                hessian[2*n+i][2*n+i] -= ((dist2[i,k] - 3*((z[i] - z[k]) ** 2)) / (dist2[i,k] ** (5/2)))
                                hessian[i][n+i] += 3 * (((x[i]-x[k])*(y[i]-y[k])) / (dist2[i,k] ** (5/2)))
                                hessian[n+i][i] += 3 * (((x[i]-x[k])*(y[i]-y[k])) / (dist2[i,k] ** (5/2)))
                                hessian[i][2*n+i] += 3 * (((x[i]-x[k])*(z[i]-z[k])) / (dist2[i,k] ** (5/2)))
                                hessian[2*n+i][i] += 3 * (((x[i]-x[k])*(z[i]-z[k])) / (dist2[i,k] ** (5/2)))
                                hessian[n+i][2*n+i] += 3 * (((y[i]-y[k])*(z[i]-z[k])) / (dist2[i,k] ** (5/2)))
                                hessian[2*n+i][n+i] += 3 * (((y[i]-y[k])*(z[i]-z[k])) / (dist2[i,k] ** (5/2)))
                    else:
                        hessian[i][j] = ((dist2[i,j] - 3*((x[i]-x[j]) ** 2)) / (dist2[i,j] ** (5/2)))
                        hessian[n+i][n+j] = ((dist2[i,j] - 3*((y[i]-y[j]) ** 2)) / (dist2[i,j] ** (5/2)))
                        hessian[2*n+i][2*n+j] = ((dist2[i,j] - 3*((z[i]-z[j]) ** 2)) / (dist2[i,j] ** (5/2)))
                        hessian[i][n+j] = -3 * ( ((x[i]-x[j])*(y[i]-y[j])) / (dist2[i,j] ** (5/2)) )
                        hessian[n+i][j] = -3 * ( ((x[i]-x[j])*(y[i]-y[j])) / (dist2[i,j] ** (5/2)) )
                        hessian[i][2*n+j] = -3 * ( ((x[i]-x[j])*(z[i]-z[j])) / (dist2[i,j] ** (5/2)) )
                        hessian[2*n+i][j] = -3 * ( ((x[i]-x[j])*(z[i]-z[j])) / (dist2[i,j] ** (5/2)) )
                        hessian[n+i][2*n+j] = -3 * ( ((y[i]-y[j])*(z[i]-z[j])) / (dist2[i,j] ** (5/2)) )
                        hessian[2*n+i][n+j] = -3 * ( ((y[i]-y[j])*(z[i]-z[j])) / (dist2[i,j] ** (5/2)) )

            return k * (ensemble_properties["charge"] ** 2) * hessian
        else:
            return None

    def force(self, positions, ensemble_properties, language="python"):
        """Computes the force on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language)

    def acceleration(self, positions, ensemble_properties, language="python"):
        """Computes the accleration on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class MutualCoulombHarmonicPotential:
    """
    Second-order approximation of potential due to Coulomb interations among all ions in the chain.
    Expansion is around the ring trap potential minimum, so first-order derivatives vanish.
    """

    def __init__(self, r0, cutoff_distance):
        """
        cutoff_distance (units: m) is the distance beyond which ion-ion Coulomb interactions are neglected for computational efficiency. 
        r0 (units: m) is the equlibrium coordinates around which to expand the potential
        """
        self.r0 = r0
        self.cutoff_distance = cutoff_distance

        n = r0.size // 3
        self.d = np.zeros((n, n))
        d2 = np.zeros((n, n))
        x = r0[:n]
        y = r0[n:2*n]
        z = r0[2*n:3*n]

        # Compute distance between each pair of ions in the chain
        for i in range(n):
            for j in range(n):
                d2[i][j] = ((x[i] - x[j]) ** 2) + ((y[i] - y[j]) ** 2) + ((z[i] - z[j]) ** 2)   
                self.d[i][j] = np.sqrt(d2[i][j])     

        # Compute matrix of second-order partial derivatives
        self.hess = np.zeros((3*n, 3*n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # same ion
                    self.hess[i][i] += k * ( ( 3*((x[i] - x[j])**2) - d2[i][j] ) / (d2[i][j] ** (5/2)) )
                    self.hess[n+i][n+i] += k * ( ( 3*((y[i] - y[j])**2) - d2[i][j] ) / (d2[i][j] ** (5/2)) )
                    self.hess[2*n+i][2*n+i] += k * ( ( 3*((z[i] - z[j])**2) - d2[i][j] ) / (d2[i][j] ** (5/2)) )
                    self.hess[i][n+i] += 3*k*( ((x[i] - x[j]) * (y[i] - y[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[n+i][i] += 3*k*( ((x[i] - x[j]) * (y[i] - y[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[n+i][2*n+i] += 3*k*( ((y[i] - y[j]) * (z[i] - z[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[2*n+i][n+i] += 3*k*( ((y[i] - y[j]) * (z[i] - z[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[2*n+i][i] += 3*k*( ((z[i] - z[j]) * (x[i] - x[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[i][2*n+i] += 3*k*( ((z[i] - z[j]) * (x[i] - x[j])) / (d2[i][j] ** (5/2)) )

                    # different ion
                    self.hess[i][j] = k * ( ( 3*((x[i] - x[j])**2) - d2[i][j] ) / (d2[i][j] ** (5/2)) )
                    self.hess[n+i][n+j] = k * ( ( 3*((y[i] - y[j])**2) - d2[i][j] ) / (d2[i][j] ** (5/2)) )
                    self.hess[2*n+i][2*n+j] = k * ( ( 3*((z[i] - z[j])**2) - d2[i][j] ) / (d2[i][j] ** (5/2)) )
                    self.hess[i][n+j] = -3*k*( ((x[i] - x[j]) * (y[i] - y[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[n+i][j] = -3*k*( ((x[i] - x[j]) * (y[i] - y[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[n+i][2*n+j] = -3*k*( ((y[i] - y[j]) * (z[i] - z[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[2*n+i][n+j] = -3*k*( ((y[i] - y[j]) * (z[i] - z[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[2*n+i][j] = -3*k*( ((z[i] - z[j]) * (x[i] - x[j])) / (d2[i][j] ** (5/2)) )
                    self.hess[i][2*n+j] = -3*k*( ((z[i] - z[j]) * (x[i] - x[j])) / (d2[i][j] ** (5/2)) )

    def potential(self, positions, ensemble_properties, language="python"):
        """
        Computes the second-order approximation of the potential due to the ions' Coulomb interactions given the coordinates of all ions in the chain.
        """
        if language == "python":
            n = ensemble_properties["n"]
            half_charge2 = 0.5 * (ensemble_properties["charge"] ** 2)
            energy = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        energy += k*(ensemble_properties["charge"]**2) / self.d[i][j]
                    energy += half_charge2 * self.hess[i][j] * (positions[i] - self.r0[i]) * (positions[j] - self.r0[j])
                    energy += half_charge2 * self.hess[n+i][n+j] * (positions[n+i] - self.r0[n+i]) * (positions[n+j] - self.r0[n+j])
                    energy += half_charge2 * self.hess[2*n+i][2*n+j] * (positions[2*n+i] - self.r0[2*n+i]) * (positions[2*n+j] - self.r0[2*n+j])
                    energy += half_charge2 * self.hess[i][n+j] * (positions[i] - self.r0[i]) * (positions[n+j] - self.r0[n+j])
                    energy += half_charge2 * self.hess[n+i][2*n+j] * (positions[n+i] - self.r0[n+i]) * (positions[2*n+j] - self.r0[2*n+j])
                    energy += half_charge2 * self.hess[2*n+i][j] *  (positions[2*n+i] - self.r0[2*n+i]) * (positions[j] - self.r0[j])
            return energy
        else:
            return cpotentials.mutual_coulomb_harmonic_potential(ensemble_properties["charge"], self.d, self.hess, self.r0, positions)
    
    def jac(self, positions, ensemble_properties, language="python"):
        """
        Computes the gradient vector of the harmonic approximation of the Coulomb potential given the coordinates of all ions in the chain.
        The returned gradient array follows the same coordinate ordering as the positions array.
        """
        if language == "python":
            n = ensemble_properties["n"]
            charge = ensemble_properties["charge"]
            grad = np.zeros(3*n)
            for i in range(n):
                for j in range(n):
                    grad[i] += (charge ** 2) * self.hess[i][j] * (positions[j] - self.r0[j]) \
                             + (charge ** 2) * self.hess[i][n+j] * (positions[n+j] - self.r0[n+j]) \
                             + (charge ** 2) * self.hess[i][2*n+j] * (positions[2*n+j] - self.r0[2*n+j])
                    grad[n+i] += (charge ** 2) * self.hess[n+i][j] * (positions[j] - self.r0[j]) \
                             + (charge ** 2) * self.hess[n+i][n+j] * (positions[n+j] - self.r0[n+j]) \
                             + (charge ** 2) * self.hess[n+i][2*n+j] * (positions[2*n+j] - self.r0[2*n+j])
                    grad[2*n+i] += (charge ** 2) * self.hess[2*n+i][j] * (positions[j] - self.r0[j]) \
                             + (charge ** 2) * self.hess[2*n+i][n+j] * (positions[n+j] - self.r0[n+j]) \
                             + (charge ** 2) * self.hess[2*n+i][2*n+j] * (positions[2*n+j] - self.r0[2*n+j])
            return grad
        else:  
            return cpotentials.mutual_coulomb_harmonic_jac(ensemble_properties["charge"], self.hess, self.r0, positions)

    def force(self, positions, ensemble_properties, language="python"):
        """Computes the force on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language)

    def acceleration(self, positions, ensemble_properties, language="python"):
        """Computes the accleration on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class PointChargePotential:
    """Potential due to a single fixed charge defect in the ring trap."""

    def __init__(self, position, charge):
        """
        position (units: m) is the (x, y, z) coordinates of the point charge defect
        charge (units: C) is the charge of the point charge defect
        """
        self.position = position
        self.charge = charge
    
    def potential(self, positions, ensemble_properties, language="python"):
        """Computes the potential due to a point charge defect given the positions of all ions in the chain."""
        if language == "python":
            n = ensemble_properties["n"]
            energy = 0
            for i in range(n):
                ri = positions[i::n]
                energy += ((ensemble_properties["charge"] * self.charge) / (4*np.pi*eps0)) * (1 / np.linalg.norm(ri - self.position))
            return energy
        else:
            return cpotentials.point_charge_potential(self.charge,
                                                      self.position,
                                                      ensemble_properties["charge"],
                                                      positions)

    def jac(self, positions, ensemble_properties, language="python"):
        """
        Computes the gradient vector of the potential due to a point charge defect given the coordinates of all ions in the chain.
        The returned gradient array follows the same coordinate ordering as the positions array.
        """
        if language == "python":
            n = ensemble_properties["n"]
            grad = np.zeros(3*n)
            for i in range(n):
                ri = positions[i::n]
                coulomb = ((ensemble_properties["charge"] * self.charge) / (4*np.pi*eps0)) * (1 / (np.linalg.norm(ri - self.position) ** 2))
                dr = (ri - self.position) / np.linalg.norm(ri - self.position)
                grad[i::n] -= coulomb * dr
            return grad
        else:
            return np.array(cpotentials.point_charge_jac(self.charge,
                                                      self.position,
                                                      ensemble_properties["charge"],
                                                      positions))

    def force(self, positions, ensemble_properties, language="python"):
        """Computes the force on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language)

    def acceleration(self, positions, ensemble_properties, language="python"):
        """Computes the accleration on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class LocalHarmonicPotential1D:
    """Simple 1d harmonic potential"""

    def __init__(self, w):
        """w (units: rad/s) is the frequency of the harmonic potential."""
        self.w = w

    def potential(self, positions, ensemble_properties, language="python"):
        """Computes the potential due to the harmonic well given the 1D positions of all the ions."""
        if language == "python":
            return 0.5 * ensemble_properties["mass"] * np.sum(self.w * self.w * positions * positions)
        else:
            return cpotentials.local_harmonic_potential_1d(ensemble_properties["mass"],
                                                           self.w,
                                                           positions
            )
    
    def jac(self, positions, ensemble_properties, language="python"):
        """Computes the gradient vector of the potential due to the harmonic well defect given the 1D coordinates of all ions in the chain."""
        if language == "python":
            return ensemble_properties["mass"] * self.w * self.w * positions
        else:
            return np.array(cpotentials.local_harmonic_jac_1d(ensemble_properties["mass"],
                                                              self.w,
                                                              positions))
    
    def force(self, positions, ensemble_properties, language="python"):
        """Computes the force on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language)
    
    def acceleration(self, positions, ensemble_properties, language="python"):
        """Computes the accleration on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class InverseSquarePotential1D:
    """Simple 1D inverse-square potential with some equilibrium distances"""

    def __init__(self, d):
        """d (units: m) is a list of equilibrium 1D coordinates of each ion."""
        self.d = d

    def potential(self, positions, ensemble_properties, language="python"):
        """Computes the inverse square potential given the 1D positions of all the of the ions."""
        if language == "python":
            n = ensemble_properties["n"]
            energy = 0
            for i in range(n):
                for j in range(i+1, n):
                    energy += (ensemble_properties["charge"] ** 2) / np.abs( (positions[i] + self.d[i]) - (positions[j] + self.d[j]) )
            return energy
        else:
            return cpotentials.inverse_square_potential_1d(self.d,
                                                           ensemble_properties["charge"],
                                                           positions)
    
    def jac(self, positions, ensemble_properties, language="python"):
        """Computes the gradient vector of the inverse-square potential given the 1D coordinates of all ions in the chain."""
        if language == "python":
            n = ensemble_properties["n"]
            grad = np.zeros(n)
            for i in range(n):
                for j in range(i+1, n):
                    mag = (ensemble_properties["charge"] ** 2) / (( (positions[i] + self.d[i]) - (positions[j] + self.d[j]) ) ** 2)
                    grad[i] += (-1 if (positions[i] + self.d[i] > positions[j] + self.d[j]) else 1) * mag
                    grad[j] += (1 if (positions[i] + self.d[i] > positions[j] + self.d[j]) else -1) * mag
            return grad
        else:
            return np.array(cpotentials.inverse_square_jac_1d(self.d,
                                                              ensemble_properties["charge"],
                                                              positions))
    
    def force(self, positions, ensemble_properties, language="python"):
        """Computes the force on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language)
    
    def acceleration(self, positions, ensemble_properties, language="python"):
        """Computes the accleration on each ion due to this potential"""
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]
