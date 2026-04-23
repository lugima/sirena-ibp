import itertools
import logging
import sys
import flint as fl
import sympy as sp
from multiprocessing import Pool, cpu_count

from .in_out_utils import propagator_matrix
from .sint_canonization import compute_shifts_dictionary
from . import global_precomputed_functions

d = sp.symbols('d')


def _wrapper_canonize_sint(sint: tuple) -> set:
    """ Canonizes a tuple of sum-integral and returns the set of canonized sum-integrals. """
    return set(list(global_precomputed_functions.canonize_sint(*sint)))

def init_worker(prop_matrix):
    global global_precomputed_functions
    from .sint_canonization import compute_shifts_dictionary

    # Set the canonize and build_ibp functions for all the modules that need it
    class DisMatrix:
        Dis_matrix = prop_matrix
    CANONIZATION_DICT, canonize_sint, cache_info = compute_shifts_dictionary(DisMatrix)

    global_precomputed_functions.set_canonize_function(canonize_sint)


def _generate_seeds_internal(loop_num: int, fer: bool, max_r: int, max_s: int, alpha_ini: int, mass_dim: set) -> list:
    """ Returns list of non-canonized seed sum-integrals at given:
    loop order, max. sum of denominator (r) and numerator powers (s), most negative denominator (alpha_ini), mass dimension and signature.
    """

    # Signatures
    if loop_num == 2:
        bf_list = [(0, 0), (1, 0), (0, 1), (1, 1)]
    elif loop_num == 3:
        bf_list = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    else:
        raise ValueError(f"{loop_num} sum-integrals are not yet implemented")

    # Number of independent propagators at each loop order
    den_combinations = loop_num * (loop_num + 1) // 2

    # Propagator powers
    alpha_list = [a for a in itertools.product(range(alpha_ini, max_r-alpha_ini+1), repeat=den_combinations)
                  if sum(a) <= max_r and sum(neg for neg in a if neg<0) >= alpha_ini
                  ]

    if alpha_ini < 0:
        # Keep seeds with sum of negative powers equal to or greater than alpha_ini
        alpha_list = [a for a in alpha_list if sum(neg for neg in a if neg<0) >= alpha_ini]

    # Numerator powers
    beta_list = [t for t in itertools.product(range(max_s+1), repeat=den_combinations)
                 if sum(t) <= max_s
                 ]
    
    if not fer:
        # Keep only bosonic signature
        bf_list = [bf_list[0]]

    seeds = ((alpha, beta, bf) for bf in bf_list 
                    for alpha in alpha_list
                    for beta in beta_list)
    
    # Keep only seeds of specified mass dimension(s)
    filtered_seeds = tuple(sint for sint in seeds 
                      if 4 * loop_num - 2 * sum(list(sint[0])) + sum(list(sint[1])) in mass_dim)

    return filtered_seeds

def generate_seeds(loop_num: int, fer: bool, max_r: int, max_s: int, alpha_ini: int, sig_order: str, mass_dim: set, n_cpus) -> list:
    """ Returns list of seed sum-integrals at given:
    loop order, max. sum of denominator (r) and numerator powers (s), most negative denominator (alpha_ini), mass dimension and signature.

    Can be parallelized by setting n_cpus to the number of cores to use, or 'auto' (by default) to use all available cores.

    Mixed numerators are removed upon canonization, so seeds are returned with zero powers in them.    
    """
    
    prop_matrix = propagator_matrix(loop_num)
    class DisMatrix:
        Dis_matrix = prop_matrix
    CANONIZATION_DICT, canonize_sint, cache_info = compute_shifts_dictionary(DisMatrix, sig_order)
    global_precomputed_functions.set_canonize_function(canonize_sint)

    # Number of independent propagators at each loop order
    den_combinations = loop_num * (loop_num + 1) // 2

    # Seeds before canonization
    seeds = _generate_seeds_internal(loop_num, fer, max_r, max_s, alpha_ini, mass_dim)

    if n_cpus == 'auto':
        n_cpus = cpu_count()
    
    if n_cpus > 1:
        logging.info(f"\nParallelizing seed generation in {n_cpus} cores...")

        # Parallelize canonization of initial list of seeds
        c_seeds = set()

        if sys.version_info >= (3, 14):
            with Pool(n_cpus, initializer=init_worker, initargs=(prop_matrix,)) as pool:
                for c_sint in pool.imap_unordered(_wrapper_canonize_sint, seeds, chunksize=1000):
                    c_seeds.update(c_sint)
        else:
            with Pool(processes=n_cpus) as pool:
                for c_sint in pool.imap_unordered(_wrapper_canonize_sint, seeds, chunksize=1000):
                    c_seeds.update(c_sint)
    
    elif n_cpus == 1:
        # Canonize initial list of seeds in single core
        init_worker(prop_matrix)
        c_seeds = {sint for seed_sint in seeds for sint in _wrapper_canonize_sint(seed_sint)}
    
    else:
        raise ValueError(f"The number of cores (n_cpus) must be a positive integer or 'auto' (got {n_cpus})")
        
    # Return canonized seeds with mixed numerators removed
    fill_zeros_beta = [0] * (den_combinations - loop_num)
    
    return [(sint[0],(*sint[1],*fill_zeros_beta), sint[2]) for sint in c_seeds]


def generate_ibp(dmom_num: int, mom_num: int, sint: tuple) -> dict:
    """ Returns the IBP equation for a given sum-integral as dictionary. 
    The dictionary represents a sum of its keys (sum-integrals), with their corresponding values (coefficients in terms of Flint integer polynomials in d).

    P, Q, R, ... correspond to 0, 1, 2, ... in both differentiated (dmom_num) and multiplied (mom_num) momenta     
    """

    # Generate IBP without including signature
    ibp = global_precomputed_functions.build_ibp(dmom_num, mom_num, (sint[0], sint[1]))

    # Add signature
    ibp = {(*k, sint[2]): v for k, v in ibp.items()}

    # Canonize
    c_ibp = {} 
    for k, v in ibp.items():
        # Remove sum-integrals with null coefficient 
        if v != 0:
            # Multiply the canonized relation times the coefficient of the sumint
            new_c_ibp = {k1: c_ibp.get(k1,0) + v*v1 for k1, v1 in global_precomputed_functions.canonize_sint(*k).items() if k1}
            c_ibp.update({k: v for k,v in new_c_ibp.items()})

    return {k: fl.fmpz_poly(list(map(int, reversed(sp.Poly(v,d).all_coeffs())))) for k, v in c_ibp.items() if v != 0}