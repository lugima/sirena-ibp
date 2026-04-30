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

def generate_betas(length, target_sum):
    """ Generates tuples of integers with numerator powers (beta)
    using "stars and bars" combinatorial trick.
    """
    
    if target_sum < 0: return

    # Base case
    if length == 1:
        yield (target_sum,)
        return
    
    # Generate combinations of integers with sum = target_sum and length = length
    for c in itertools.combinations(range(target_sum + length - 1), length - 1):
        # Compute numbers of stars (b-a-1) between each pair of bars (a, b) for each 
        # combinations of betas (c) --- All different ways to arrange the betas
        yield tuple(b - a - 1 for a, b in zip((-1,) + c, c + (target_sum + length - 1,)))


def generate_alphas(length, target_sum, alpha_ini, max_val, neg_allowance):
    """ Recursively generates tuples of integers with propagator powers (alpha).
    'target_sum' is the sum of the tuple, and each value lies between 'alpha_ini' and 'max_val'.
    'neg_allowance' ensures the sum of negative powers is never below 'alpha_ini'.
    """

    # Base case: if there are no elements left to choose, yield an empty tuple or stop
    if length == 0:
        if target_sum == 0:
            yield ()
        return
    
    for val in range(alpha_ini, max_val + 1):
        # Update negative allowance
        new_allowance = neg_allowance + val if val < 0 else neg_allowance
        
        if new_allowance >= 0:
            rem_len = length - 1
            rem_target = target_sum - val
            
            # Prune branches where the sum of alpha is unreachable
            min_possible = -new_allowance
            max_possible = rem_len * max_val
            
            if min_possible <= rem_target <= max_possible:
                for tail in generate_alphas(rem_len, rem_target, alpha_ini, max_val, new_allowance):
                    yield (val,) + tail


def _wrapper_canonize_sint(sint: tuple) -> set:
    """ Canonizes a tuple of sum-integral and returns the set of canonized sum-integrals. """
    return set(list(global_precomputed_functions.canonize_sint(*sint)))


def init_worker(prop_matrix, sig_order, n_cpus):
    global global_precomputed_functions
    from .sint_canonization import compute_shifts_dictionary

    # Set the canonize and build_ibp functions for all the modules that need it
    class DisMatrix:
        Dis_matrix = prop_matrix
    CANONIZATION_DICT, canonize_sint, cache_info = compute_shifts_dictionary(DisMatrix, sig_order, n_cpus)

    global_precomputed_functions.set_canonize_function(canonize_sint)


def _generate_seeds_internal(loop_num: int, fer: bool, max_r: int, max_s: int, alpha_ini: int, mass_dim: set) -> list:
    """ Returns list of non-canonized seed sum-integrals at given:
    loop order, max. sum of denominator (r) and numerator powers (s), most negative denominator (alpha_ini), mass dimension and signature.
    """

    # Number of independent propagators at each loop order
    den_combinations = loop_num * (loop_num + 1) // 2

    # Signatures
    bf_list = list(itertools.product(range(2), repeat=loop_num)) if fer else [(0,) * loop_num]

    # Upper limit for individual value of alpha
    max_alpha_val = max_r - alpha_ini 

    filtered_seeds_list = []

    for dim in mass_dim:
        K = 4 * loop_num - dim
        
        # Iterate for all sums of beta
        for s_beta in range(0, max_s + 1):
            
            # Obtain sum of alpha for this sum of beta: 2 * sum(alpha) = K + sum(beta)
            s_alpha_2 = K + s_beta
            
            # Must be divisible by 2
            if s_alpha_2 % 2 == 0:
                s_alpha = s_alpha_2 // 2
                
                # Check sum(alpha) is within specified bounds
                if alpha_ini <= s_alpha <= max_r:
                    
                    # Generate tuples
                    valid_betas = list(generate_betas(den_combinations, s_beta))
                    allowance = -alpha_ini if alpha_ini < 0 else 0
                    valid_alphas = list(generate_alphas(den_combinations, s_alpha, alpha_ini, max_alpha_val, allowance))
                    
                    # Add signatures
                    if valid_alphas and valid_betas:
                        for bf in bf_list:
                            for a in valid_alphas:
                                for b in valid_betas:
                                    filtered_seeds_list.append((a, b, bf))

    filtered_seeds = tuple(filtered_seeds_list)

    return filtered_seeds


def generate_seeds(loop_num: int, fer: bool, max_r: int, max_s: int, alpha_ini: int, sig_order: str, mass_dim: set, n_cpus) -> list:
    """ Returns list of seed sum-integrals at given:
    loop order, max. sum of denominator (r) and numerator powers (s), most negative denominator (alpha_ini), mass dimension and signature.

    Can be parallelized by setting n_cpus to the number of cores to use, or 'auto' (by default) to use all available cores.

    Mixed numerators are removed upon canonization, so seeds are returned with zero powers in them.    
    """
    
    # Build canonization dictionary
    prop_matrix = propagator_matrix(loop_num)
    class DisMatrix:
        Dis_matrix = prop_matrix
    CANONIZATION_DICT, canonize_sint, cache_info = compute_shifts_dictionary(DisMatrix, sig_order, n_cpus)
    global_precomputed_functions.set_canonize_function(canonize_sint)

    # Number of independent propagators at each loop order
    den_combinations = loop_num * (loop_num + 1) // 2

    # Seeds before canonization
    seeds = _generate_seeds_internal(loop_num, fer, max_r, max_s, alpha_ini, mass_dim)

    if n_cpus == 'auto':
        n_cpus = cpu_count()
    
    if n_cpus > 1:
        logging.info(f"\nParallelizing seed canonization in {n_cpus} cores...")

        # Parallelize canonization of initial list of seeds
        c_seeds = set()

        if sys.version_info >= (3, 14):
            with Pool(n_cpus, initializer=init_worker, initargs=(prop_matrix,sig_order,n_cpus)) as pool:
                for c_sint in pool.imap_unordered(_wrapper_canonize_sint, seeds, chunksize=1000):
                    c_seeds.update(c_sint)
        else:
            with Pool(processes=n_cpus) as pool:
                for c_sint in pool.imap_unordered(_wrapper_canonize_sint, seeds, chunksize=1000):
                    c_seeds.update(c_sint)
    
    elif n_cpus == 1:
        # Canonize initial list of seeds in single core
        init_worker(prop_matrix, sig_order, n_cpus)
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