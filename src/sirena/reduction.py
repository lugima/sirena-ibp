from multiprocessing import Pool, cpu_count

import numpy as np
import sympy as sp
import flint as fl

from .ibp_generation import pre_build_ibp_relations
from .seed_generation import generate_seeds, generate_ibp
from . import global_precomputed_functions
from .ibp_system_handling import *
from .ibp_sols_handling import *
from .sint_utils import *
from .in_out_utils import *

d = sp.symbols('d')


def sirena(sints, max_r=6, max_s=6, alpha_ini=0, sig_order="normal", n_cpus=['auto',1], basis_sints=[]):
    """ Finds the IBP reduction of an input set of sum-integrals
    
    For seed generation, must specify: max. sum of denominator (r) and numerator powers (s). 
    Can also specify a list of sum-integrals (basis_sints) to prioritize as masters

    Can be parallelized by setting n_cpus to the number of cores to use, or 'auto' (by default) to use all available cores. 
    n_cpus[0] are used for seed generation and n_cpus[1] are used for solving the IBP system
    """

    if len(n_cpus) != 2:
        raise ValueError(f"n_cpus must contain two elements, but {len(n_cpus)} were provided.")

    # Convert to tuple single sum-integral inputs
    if type(sints) != tuple:
        sints = tuple(sints)

    loop_num = len(sints[0][2])
    fer = are_fermionic(sints)
    mass_dim = get_mass_dim(sints)

    if loop_num > 2 and fer and (max_r > 6 or alpha_ini < 0):
        print_cafe()

    logging.info(" ")
    logging.info("-"*50)
    logging.info("Computing IBPs for seed sum-integrals...")
    logging.info("-"*50)

    prop_matrix = propagator_matrix(loop_num)
    build_ibp = pre_build_ibp_relations(loop_num, prop_matrix)
    global_precomputed_functions.set_build_ibp_function(build_ibp)

    seed_input = generate_seeds(loop_num=loop_num,
                fer=fer,
                max_r=max_r,
                max_s=max_s,
                alpha_ini=alpha_ini,
                sig_order=sig_order,
                mass_dim=mass_dim,
                n_cpus=n_cpus[0])
    
    masters_count_pre, masters_count = 0, 1
    counter = 0
    already_computed_ibps = []

    logging.info(" ")
    logging.info("-"*50)
    logging.info("Reducing number of master integrals in solution...")
    logging.info("-"*50)

    # Repeat numerical solution adding neighboring seeds until number of masters stops decreasing
    while masters_count != masters_count_pre:

        counter += 1
        masters_count_pre = masters_count

        logging.info(" ")
        logging.info("=" * 12)
        logging.info(f"Run number {counter}")
        logging.info("=" * 12)   
        
        sols, extra_info = solve_from_seeds(
            sints, 
            seed_input,
            loop_num=loop_num,
            basis_sints=basis_sints, 
            already_computed_ibps=already_computed_ibps,
            return_extra_info=True,
            numerical=True,
            n_cpus=n_cpus[1]
            )
        
        # If number of generated IBPs is the same as before, its solution will not change
        if sols is None:
            break
        
        masters = set(sint for eq in sols for sint in eq)
        masters_count = len(masters)

        logging.info(f"\n>> Found solution in terms of {masters_count} master sum-integrals <<\n\n")

        # Save IBPs to avoid re-generating them in next iteration
        already_computed_ibps = extra_info["ibps"]
        seed_input_new = list(masters) + neighbor_sints(masters)
        seed_input_new = list({fill_zero_indices(c_sint)
                           for sint in seed_input_new 
                           for c_sint in global_precomputed_functions.canonize_sint(*fill_zero_indices(sint))})
        
        # Keep only new neighboring seeds
        seed_input = list(set(seed_input_new) - set(seed_input))

    # Solve one last time with polynomials in d
    logging.info(" ")
    logging.info("-"*50)
    logging.info(f"Solving IBP system symbolically...")
    logging.info("-"*50)


    final_sols = solve_from_seeds(
            sints, 
            seed_input,
            loop_num=loop_num,
            basis_sints=basis_sints, 
            already_computed_ibps=already_computed_ibps,
            return_extra_info=False,
            numerical=False,
            n_cpus=n_cpus[1]
            )

    return final_sols


def _wrapper_solve_system(decoup_ibp_w_numerical):
    decoup_ibp, numerical = decoup_ibp_w_numerical
    # Reconvert list of coefficients to Flint polys for solving (after pickling in multiprocessing)
    decoup_ibp = [{sint : fl.fmpz_poly(list(poly)) for sint, poly in eq.items()} for eq in decoup_ibp]
    solution = solve_system(decoup_ibp, numerical, verbose=False)
    # If symbolic, convert Flint polys to list of coefficients for pickling again
    if not numerical:
        solution = [(tag, {sint : [int(c) for c in poly.get_coeffs()] for sint, poly in eq.items()}) for tag, eq in solution]
    return solution


def solve_from_seeds(sints, seeds, loop_num, already_computed_ibps=[], basis_sints=[], 
                     return_extra_info=False, numerical=False, n_cpus='auto'):
    """ Finds the IBP reduction of an input set of sum-integrals, given a set of seeds """

    # Add sints to seeds and canonize
    extra_seeds = set(sints)
    extra_seeds = {fill_zero_indices(c_sint) 
                        for sint in extra_seeds 
                        for c_sint in global_precomputed_functions.canonize_sint(*fill_zero_indices(sint))}
    seeds = list(set(seeds) | extra_seeds)

    # Generate IBPs
    ibps = [generate_ibp(dmom_num, mom_num, sint) 
                for sint in seeds 
                for dmom_num, mom_num in np.ndindex((loop_num, loop_num))]
    ibps += already_computed_ibps

    # Remove duplicates and empty IBPs
    result = []
    for eq in ibps:
        if eq and eq not in result:
            result.append(eq)
    ibps = [dict(eq) for eq in result]

    logging.info(f"Number of generated IBPs: {len(ibps)}\n")

    # Return if no new IBPs were generated in the finite field iteration
    if numerical:
        if len(ibps) == len(already_computed_ibps):
            if return_extra_info:
                return None, None
            else:
                return None

    logging.info(" ")
    logging.info("-"*50)
    logging.info("Solving IBP system numerically...")
    logging.info("-"*50)

    # Tag sum-integrals by complexity 
    tagged_ibps, tag_map = tag_sints(ibps, basis_sints=basis_sints)

    # Sort IBPs
    sorted_ibps = sort_ibps(tagged_ibps)

    # Decouple IBP system in subsystems
    decoup_ibps = decouple_ibps(sorted_ibps)

    logging.info(f"Found {len(decoup_ibps)} subsystems with numbers of equations:\n{[len(decoup) for decoup in decoup_ibps]}\n")

    # Solve subsystems
    if n_cpus == 'auto':
        n_cpus = cpu_count()

    if n_cpus > 1:

        logging.info(f"\nParallelizing system solving in {n_cpus} cores...")

        # Sort subsystems by number of equations to optimize parallelization
        decoup_ibps = sorted(decoup_ibps, key=lambda x: len(x), reverse=True)
        # Convert Flint polys to list of coefficients for pickling (so that multiprocessing can be used)
        decoup_ibps = [[{sint : tuple(int(c) for c in poly) for sint, poly in eq.items()} for eq in decoup_ibp] for decoup_ibp in decoup_ibps]
        decoup_ibps_w_numerical = [(decoup_ibp, numerical) for decoup_ibp in decoup_ibps]
        with Pool(n_cpus) as pool:
            solutions = [sol for sol in pool.imap_unordered(_wrapper_solve_system, decoup_ibps_w_numerical, chunksize=1)]

    elif n_cpus == 1:

        solutions = []
        for i, decoup_ibp in enumerate(decoup_ibps):

            logging.detailed_info("\n")
            logging.detailed_info("=" * 50)
            logging.detailed_info(f"Solving subsystem {i+1} / {len(decoup_ibps)} with {len(decoup_ibp)} equations")
            logging.detailed_info("-" * 50)

            verbose = True if logging.getLogger().level == logging.DETAILED_INFO else False
            solutions.append(solve_system(decoup_ibp, numerical, verbose=verbose))

            logging.detailed_info(" ")
            logging.detailed_info("=" * 50)

    else:
        raise ValueError(f"The number of cores (n_cpus) must be a positive integer or 'auto' (got {n_cpus})")
    
    # Flatten
    solutions = [sol for solution in solutions for sol in solution]    

    if not numerical:
        # Convert Flint polynomials to sp.Poly
        if n_cpus == 1:
            ibp_solutions = [(tag, 
                        {sint: sp.Poly.from_list(list(reversed([int(c) for c in poly.get_coeffs()])),d) 
                        for sint, poly in eq.items()}) 
                        for tag, eq in solutions]
        else:
            ibp_solutions = [(tag, 
                        {sint: sp.Poly.from_list(list(reversed(poly)), d) 
                        for sint, poly in eq.items()}) 
                        for tag, eq in solutions]
    else:
        # Convert int to sp.Poly
        ibp_solutions = [(tag, 
                        {sint: sp.Poly(coeff, d) 
                        for sint, coeff in eq.items()}) 
                        for tag, eq in solutions]

    # Solve IBP system for specified sum-integrals
    sols = [solve_system_for_sint(sint, ibp_solutions, tag_map) for sint in sints]

    if return_extra_info:
        return (sols, {
            "ibp_solutions" : ibp_solutions,
            "tag_map" : tag_map,
            "ibps" : ibps
        })
    else:
        return sols
    

