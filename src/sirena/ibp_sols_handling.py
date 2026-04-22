import sympy as sp
from . import global_precomputed_functions


def solve_eq_for_sint(sint: tuple, eq: dict) -> dict:
    """ Solves equation for a given sum-integral """

    coeff = -eq[sint]
    result = {i : sp.factor(sp.cancel(c / coeff)) for i, c in eq.items() if i != sint}

    return result


def solve_system_for_sint(sint: tuple, solutions: list, tag_map: dict, canon=False) -> dict:
    """ Retrieves IBP reduction for a given sum-integral from the solved IBP system """

    if not canon:
        c_sint = global_precomputed_functions.canonize_sint(*sint)

        if not c_sint:
            return {}

        # For each term in the canonized input, get its IBP reduction
        result = {}
        for c_i, coeff in c_sint.items():
            if coeff != 0:

                term_solution = solve_system_for_sint(c_i, solutions, tag_map, canon=True)

                prov_result = {k1: sp.factor(sp.cancel(result.get(k1,0) + coeff * v1)) for k1, v1 in term_solution.items() if k1}

                result.update(prov_result)     

        return {k: v for k, v in result.items() if v != 0}

    # Find tag of input sum-integral
    tag_sint = -1
    for tag, i in tag_map.items():
        if i == sint:
            tag_sint = tag
            break

    # Return if the sum-integral is not in the tagged list
    if tag_sint == -1:
        raise ValueError(f"The sum-integral {sint} was not found in the IBP system!")
    
    # Find what's the IBP equation for the target sum-integral
    eq = {}
    for mc_sint, ibp in solutions:
        if mc_sint == tag_sint:
            eq = ibp

    # If there is no equation for it, it's a master
    if eq == {}:
        return {sint : 1}
    
    # If it's only the target in its equation, it is zero
    if len(eq) == 1:
        return {sint : 0}
    
    # Solve the equation for the target sum-integral
    result = solve_eq_for_sint(tag_sint, eq)

    # Convert tag back to list of indices
    result = {tag_map[i] : c for i, c in result.items()}

    return {k: v for k, v in result.items() if v!=0}