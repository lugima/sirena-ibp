import random
import sympy as sp
from numba import njit
import ibp_solver
import ibp_solver_num

# region Complexity sorting

def sort_sints(sints: list) -> list:
    """ Returns sorted list of sum-integrals by Laporta's complexity criterion, most complex first """

    # Split denominator and numerator indices. Also unpack (and ignore) signature
    dens, nums, _ = map(list, zip(*sints))

    sints_t = [sum(1 for i in den if i) for den in dens]    # Number of denominators
    sints_r = [sum(i for i in den) for den in dens]         # Sum of denominator powers
    sints_s = [sum(i for i in num) for num in nums]         # Sum of numerator powers (temporal momenta)
    sints_amax = [max(i for i in den) for den in dens]      # Largest denominator power
    sints_bmax = [max(i for i in num) for num in nums]      # Largest denominator power

    assigned_sints = [((sints_t[i], sints_r[i], sints_s[i], sints_amax[i], sints_bmax[i]), sint) for i, sint in enumerate(sints)]
    
    # Sort list putting larger values of each criterion first, in order
    assigned_sints.sort(key=lambda x: x[0], reverse=True)

    return [sint for _, sint in assigned_sints]


def tag_sints(ibps: list, basis_sints = []) -> list:
    """ Returns IBP set with sum-integrals tagged according to complexity. Also returns the tag - sum-integral mapping as dict

    Lower numbers correspond to higher complexity
    """

    # Find all different sum-integrals in IBP system
    indep_sints = list({sint for ibp in ibps for sint in list(ibp)})
    
    # Sort by decreasing complexity
    sorted_sints = sort_sints(indep_sints)

    sorted_sints = [si for si in sorted_sints if si not in basis_sints]
    sorted_sints += basis_sints

    # Define tagging dictionary
    tag_map = {sint: tag for tag, sint in enumerate(sorted_sints)}

    # Rename all sum-integral keys as numerical tags
    return (
        [
            {tag_map.get(sint, sint): coeff for sint, coeff in ibp.items()}
            for ibp in ibps
        ]
        ,
        {v : k for k, v in tag_map.items()}
        )


def sort_ibps(ibps: list) -> list:
    """ Sorts equations in tagged IBP system by:

    1) Most complex sum-integral
    2) Largest number of sum-integrals
    3) Second most complex sum-integral 
    4) Third most complex sum-integral
    5) etc.

    The order is reversed: least complex equations go first
    """

    decorated = []
    for ibp in ibps:

        # Sort keys (sum-integrals) in descending complexity (lower numbers first)
        keys_desc = sorted(ibp.keys())

        # Build sorting key
        decorated.append(((keys_desc[0], len(keys_desc), *keys_desc[1:]), ibp))

    # Sort by the key in ascending order
    decorated.sort(key=lambda x: x[0], reverse=True)

    return [sorted_ibp for _, sorted_ibp in decorated]

# endregion

# region Numerical evaluation

@njit
def eval_poly(coeffs: list, d_val: int, p: int):
    """ Applies Horner method for polynomial evaluation (mod p) """

    res = 0
    # For each coefficients, from a_n to a_0
    for c in coeffs:
        res = (res * d_val + c) % p
    return res


def eval_ibp(ibp: dict, d_val: int, p: int) -> int:
    """ Evaluates all polynomial coefficients in IBP dictionary (mod p) """

    return {sint: 
        eval_poly(tuple(reversed([int(c) for c in value])), d_val, p)
        for sint, value in ibp.items()}

# endregion


# region System reduction

def decouple_ibps(ibps: list) -> list:
    """ Decouples system of IBPs in independent subsystems (not sharing variables) """

    # Initialize: each set contains variables and positions
    sets = [{'vars': set(eq), 'rows': {i}} for i, eq in enumerate(ibps)]

    merged = True
    # Iterate until no equations were merged in previous step, as a merge can cause a new possible merge in next iteration
    while merged:
        merged = False
        new_sets = []
        # Iterate until there are not equations left unassigned to a subsystem
        while sets:
            first, *rest = sets
            first = dict(first)  # Copy
            rest2 = []
            for s in rest:
                # If variables intersect
                if first['vars'] & s['vars']:
                    # Merge variables and rows
                    first['vars'] |= s['vars']
                    first['rows'] |= s['rows']
                    merged = True
                else:
                    rest2.append(s)
            new_sets.append(first)
            # Equations unassigned to a subsystem return to the original set
            sets = rest2
        # In the next step, we try again to merge the different subsystem that were created
        sets = new_sets

    # Rows in each subsystem must be re-sorted in order to keep them sorted after merging
    return [
        [ibps[row] for row in sorted(subsyst["rows"])]
        for subsyst in sets
    ]


def is_redundant(row: dict, basis: list, p: int):
    """ Determines whether a row is a linear combination of a basis of other rows in a system of equations
        
    The approach is numerical and employs modular arithmetics (mod p)

    Returns a boolean indicating whether the row is independent, and in such case, the row and the new pivot
    """

    row = row.copy()

    # While the row has not been removed
    while row:
        # Pick leftmost non-zero column in row
        pivot = min(row)

        # If no there are no common pivots, it is independent
        if pivot not in basis:
            return False, row, pivot

        # Eliminate pivot using basis row
        factor = row[pivot]
        pivot_row = basis[pivot]
        for j, v in pivot_row.items():
            row[j] = (row.get(j, 0) - factor * v) % p

            # Eliminate row if it was removed by the operation before
            if row[j] == 0:
                del row[j]

    return True, None, None


def find_redundants(system: list, p: int) -> list:
    """ Finds all linearly-dependent rows in a system of equations

    The approach is numerical and employs modular arithmetics (mod p)

    Returns list of positions of redundant rows in system
    """

    basis = {}
    redundants = []
    for i, row in enumerate(system):
        redundant_flag, reduced_row, pivot = is_redundant(row, basis, p)

        if redundant_flag:
            redundants.append(i)
        else:
            inv = pow(reduced_row[pivot], -1, p) # Normalize pivot to 1
            for j in reduced_row:
                reduced_row[j] = (reduced_row[j] * inv) % p
            basis[pivot] = reduced_row

    return redundants


def remove_redundants(ibps: list, n_attempts: int = 3) -> list:
    """ Removes all linearly-dependent rows in a system of equations """

    # Checks if redundant rows are the same for n_attempts different pairs of values of d and the prime modulus p
    dim_list = [random.randint(1000, 10000) for i in range(n_attempts)]
    mod_list = [sp.nextprime(random.randint(10000, 100000)) for i in range(n_attempts)]

    redundant_positions = range(len(ibps))
    for dim, mod in zip(dim_list, mod_list):
        
        # Evaluate numerically all IBPs
        ibps_num = [eval_ibp(ibp, dim, mod) for ibp in ibps]
        # Find redundant rows positions
        redundant_positions = [pos for pos in find_redundants(ibps_num, mod) if pos in redundant_positions]

    return [ibp for i, ibp in enumerate(ibps) if i not in redundant_positions]


def solve_system(ibps, numerical: bool = False, verbose: bool = True):
    """ Solves IBP system, first removing linearly-dependent equations

    If numerical = True, it solves the system for random finite field with fixed modulus
    """

    reduced_ibps = remove_redundants(ibps)

    # Calls C++ solver
    if not numerical:
        eqs = [{k: ibp_solver.FlintPoly(list(v)) for k, v in eq.items()} for eq in reduced_ibps]
        return ibp_solver.solve_ibps(eqs, verbose)
    else:
        mod = 1000003 # Same modulus as in ibp_solver_num
        d_val = random.randint(1000, mod-1)
        eqs_num =  [{k: eval_poly(tuple(reversed([int(c) for c in v])), d_val, mod) 
                     for k, v in eq.items()} 
                     for eq in reduced_ibps]
        return ibp_solver_num.solve_ibps(eqs_num, verbose)

# endregion


