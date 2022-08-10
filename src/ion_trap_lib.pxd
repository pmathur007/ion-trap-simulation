from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "ion_trap_lib.h":
    double total_energy(
        const int n,
        const vector[double]& x
    );
    double neg_grad_energy(
        const int n,
        vector[double]& x,
        const int k,
        const double dx
    );
    double force(
        const int n,
        vector[double]& x,
        const int k
    );
    pair[vector[vector[double]], vector[vector[double]]] sim_leapfrog(
        const int n,
        const double T,
        const double dt,
        const double M,
        vector[double]& x_0,
        vector[double]& v_0,
        const double dx
    );
    double a_dless(
        const int n,
        vector[double]& x, 
        const int k
    );
    pair[vector[vector[double]], vector[vector[double]]] sim_leapfrog_dless(
        const int n,
        double T,
        double dt, 
        vector[double]& x_0,
        vector[double]& v_0
    );
    