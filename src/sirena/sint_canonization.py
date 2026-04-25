import numpy as np
import itertools
import scipy
import multiprocessing
import logging
from operator import itemgetter
from collections import defaultdict
from functools import cache, partial

###############################################################################
# The main and only function of this script is to compute
# the dictionary for shifts canonizations given a DisMatrix class.
# It returns the dictionary and the function ready to canonize any sum-integral.
#
# The DisMatrix class is just a class with one attribute, corresponding to
# the coefficient matrix of the denominators of the sum-integral, i.e.,
# class DisMatrix:
#       Dis_matrix = np.array([
#           [1, 0],
#           [0, 1],
#           [1, -1]
#       ])
###############################################################################

# region Auxiliary

def permute_rows(M, perm):
    """ Permutes rows of a matrix given a permutation.

    The permutation of n rows takes the form [i_1 i_2 ... i_n]
    """
    M_ = M.copy()
    for i, p in enumerate(perm):
        M_[i,:] = M[p,:]
    return M_


def has_solution_fast(NT, B):
    """ Decides whether there exists solution for a system like M @ X = D @ B, where X is the unknown
    and D=diag(d_i) is whatever diagonal matrix with d_i=+-1.

    The input is the transpose of the kernel of M (NT) and B.

    It returns two outputs:
        - Whether there exists solution or not (bool)
        - The values d_i
    """

    C = np.einsum('ij,jk->ijk', NT, B)
    C = C.transpose(0, 2, 1).reshape(-1, C.shape[1])

    # The vectors of NC are the possible values d_i of D=diag(d_i)
    # such that M @ X = D @ B has solution for X
    NC = scipy.linalg.null_space(C).T

    # Is any +- 1 vector in NC?
    if NC.shape[0] == 1:
        if np.any(np.isclose(NC[0], 0.0)):
            return (False, "")
        else:
            return (True, NC[0]/NC[0,0])
    elif NC.shape[0] > 1:
        NC_kernel = scipy.linalg.null_space(NC).T
        for signs in itertools.product([-1,1], repeat=NC.shape[1]):
            signs = np.array(signs)
            if np.allclose(NC_kernel @ signs, 0):
                return (True, signs)
        return (False, "")
    else:
        return (False, "")


def find_shift(NT, PINV, B, tol=1e-12):
    """ Finds the matrix X for which M @ X = D @ B, where
    D=diag(d_i) is whatever diagonal matrix with d_i=+-1.

    The input is the transpose of the kernel of M (NT), 
    the pseudoinverse of M (PINV) and B.

    It returns the value for X
    """

    # Check if there exists solution for some D and capture this D
    exists_sol, diag = has_solution_fast(NT, B)
    if not exists_sol:
        return 
    
    shift = PINV @ (B * diag.reshape(-1, 1))
    shift[np.abs(shift) < tol] = 0

    return shift


def invert_matrix(m):
    """ Inverts a matrix, giving zero if the matrix is singular. """
    try:
        return np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return np.zeros_like(m)


@cache # This is the heaviest function. Hence, it is cached.
def expand_fast(matrix, alpha):
    """ Multinomial expands a matrix M raised to some powers.
    e.g.,
    M = [
        [1 0 0],
        [0 1 0],
        [1 0 -1],
        [1 2 0]
    ]

    alpha = [1 2 3 4]

    The result is the dictionary of coefficients of the polynomial in x1, x2, x3 resulting from
    (1*x1 + 0*x2 + 0*x3)^1 * (0*x1 + 1*x2 + 0*x3)^2 * (1*x1 + 0*x2 - 1*x3)^3 * (1*x1 + 2*x2 + 0*x3)^4
    """

    M = np.array(matrix)
    k = len(M[0])

    # Initializing result
    poly = {tuple([0]*k): 1}

    for row, power in zip(M, alpha):

        # Power expansion of a row (iterative)
        part = defaultdict(int)
        part[(0,)*k] = 1

        for _ in range(power):
            new = defaultdict(int)
            for exp, coeff in part.items():
                for j, a in enumerate(row):
                    if a == 0:
                        continue
                    e = list(exp)
                    e[j] += 1
                    new[tuple(e)] += coeff * a
            part = new

        # Multiplication of all the polynomials
        new_poly = defaultdict(int)
        for e1, c1 in poly.items():
            for e2, c2 in part.items():
                e = tuple(a+b for a,b in zip(e1, e2))
                new_poly[e] += c1 * c2

        poly = new_poly

    return dict(poly)


def get_sector(alpha):
    """ Yields the binary representation of the sector with indices alpha. """
    return ''.join(['1' if a != 0 else '0' for a in alpha])


def chop_to_int(x, tol=1e-7):
    """ Chops a number to the nearest integer. """
    r = round(x)
    return r if abs(x - r) <= tol else x


def process_permutation_worker(p, dis_matrix, filter_indices, N, PINV):
    """ Worker function to process a single permutation. """
    B = permute_rows(dis_matrix, p)
    B = B[filter_indices]
    
    shift = find_shift(N, PINV, B)
    
    if shift is not None:
        return (p, shift)  # Return both so we can unpack them later
    return None

# endregion


def compute_shifts_dictionary(DisMatrix, sig_order="normal", n_cpus=1):

    if n_cpus == 'auto':
        n_cpus = multiprocessing.cpu_count()

    # region Sum-integral classes

    class SumIntegralSuperSector(DisMatrix):

        _instances = {}  # instances_dictionary to make it a singleton

        # Handling the creation of new instances to make it a singleton
        def __new__(cls, n):
            if n in cls._instances:
                return cls._instances[n]  # return instance if already created
            instance = super().__new__(cls)  # create new instance
            cls._instances[n] = instance
            return instance


        # Initialization of the instance
        def __init__(self, supersector):
            self.supersector = supersector

            # List of all possible binary representation of sectors within this supersector
            rows = list(itertools.combinations(range(len(self.Dis_matrix)), supersector))
            self.binary_sectors = []
            for row in rows:
                binary_sector = [0 if i not in row else 1 for i in range(len(self.Dis_matrix))]
                self.binary_sectors.append(''.join(map(str, binary_sector)))
            self.binary_sectors.sort(reverse=True)


        def find_physical_sectors(self):
            """ Finds all physical sectors and their correspondence within this supersector.

                Workflow
                --------
                
                1. Start with the list of all binary sectors
                2. Pick the first sector from the list
                   This list is ordered, so that when it picks one sector, it always picks the canonical one
                3. Create a SumIntegralSector instance for that sector
                   This will automatically find all related sectors and their permutations
                4. Store the canonical sector and its related sectors in a dictionary
                   Also store all permutations 
                5. Remove all related sectors from the list of binary sectors
                6. Repeat until no sectors are left in the list
            """

            self.physical_sectors = {}
            self.permutations_to_canonical = {} 
            self.shifts_to_canonical = {} 

            sector_list = self.binary_sectors.copy()

            counter = 0
            while len(sector_list) > 0:
                sector = SumIntegralSector(sector_list[counter])
                self.physical_sectors[sector.canonical_sector] = sector.related_sectors
                
                sector_list = [s for s in sector_list if s not in sector.related_sectors]

                # We store permutation found from 'canonical_sector' to 'non_canonical_sector'
                # as if it were the inverse permutation. We store the inverse shifts
                self.permutations_to_canonical.update(sector.perms_to_sectors)
                dict_perms_shifts = dict(zip(sector.possible_permutations, sector.shifts))
                self.shifts_to_canonical.update({
                    sector: {perm: invert_matrix(dict_perms_shifts[perm]) for perm in perms}
                    for sector, perms in sector.perms_to_sectors.items()
                })
                
                logging.info(f"\nFound {len(sector.related_sectors)} sectors related to {sector.canonical_sector}")

            return self.permutations_to_canonical, self.shifts_to_canonical
        

    class SumIntegralSector(DisMatrix):

        _instances = {}  # cache dictionary

        def __new__(cls, n):
            if n in cls._instances:
                return cls._instances[n]  # returns the instance already created
            instance = super().__new__(cls)  # creates new instance
            cls._instances[n] = instance
            return instance


        def __init__(self, sector):

            self.sector = sector
            self.filter = [c == '1' for c in sector]

            self.possible_permutations, self.shifts = self.find_permutations()

            # Dictionary relating every equivalent sector and all the permutations reaching them (sector mappings)
            perms_to_sectors = {perm : self.permute_sector(perm) for perm in self.possible_permutations}
            self.perms_to_sectors = defaultdict(list)
            for k, v in perms_to_sectors.items():
                self.perms_to_sectors[v].append(k)
            self.perms_to_sectors = dict(self.perms_to_sectors)

            # Dictionary relating every equivalent permutation and the corresponding shift
            dict_perms_shifts = dict(zip(self.possible_permutations, self.shifts))
            self.dict_perms_shifts = {
                sector: {perm: dict_perms_shifts[perm] for perm in perms}
                for sector, perms in self.perms_to_sectors.items()
            }

            # List of all equivalent sectors
            self.related_sectors = sorted(list(self.perms_to_sectors.keys()), reverse=True)

            # Canonical sector and permutations to reach it
            self.canonical_sector = self.related_sectors[0]
            self.permutations_to_canonical_sector = self.perms_to_sectors[self.canonical_sector]


        def permute_sector(self, perm):

            res = [''] * len(self.sector)
            for i in range(len(self.sector)):
                res[perm[i]] = self.sector[i]

            return ''.join(res)


        def find_permutations(self):

            # Matrix of the selected denominators
            M = self.Dis_matrix[self.filter]
            
            # Kernel of the matrix formed by the selected denominators
            N = scipy.linalg.null_space(M.T).T
            PINV = np.linalg.pinv(M)

            # All possible permutations of all denominators
            perms = list(itertools.permutations(range(self.Dis_matrix.shape[0])))

            def mod_out_permutations(perms, boolean_filter):
                """ Mods out permutations which only shuffle propagators that do not belong to the sector,
                 picking one representative for each class.
                """

                # Convert bool mask to integers
                indices = [i for i, keep in enumerate(boolean_filter) if keep]

                if len(indices) >= len(perms[0]) - 1:
                    return perms
                
                seen = set()
                seen_add = seen.add  # Localize method
                unique_perms = []
                unique_append = unique_perms.append
                
                if len(indices) == 1:
                    # If there is only one True in filter, itemgetter does not return tuple so we do it manually
                    idx = indices[0]
                    for p in perms:
                        key = p[idx]
                        if key not in seen:
                            seen_add(key)
                            unique_append(p)
                            
                elif len(indices) > 1:
                    get_keys = itemgetter(*indices)
                    for p in perms:
                        key = get_keys(p)
                        if key not in seen:
                            seen_add(key)
                            unique_append(p)
                            
                else:
                    # If filter is full False (should not happen)
                    return [perms[0]] if perms else []
                    
                return unique_perms

            unique_perms = mod_out_permutations(perms, self.filter)

            paral_threshold = 10000 # Number of permutations beyond which we parallelize
            chunksize = 1000 # Number of permutations to apply in each chunk passed to a core

            # Parallelize if there are many permutations to apply
            if n_cpus > 1 and len(unique_perms) > paral_threshold:
                with multiprocessing.Pool(processes=n_cpus) as pool:
                    worker_func = partial(process_permutation_worker, dis_matrix=self.Dis_matrix, filter_indices=self.filter, N=N, PINV=PINV)
                    results = pool.map(worker_func, unique_perms, chunksize=chunksize)

                perms_solutions = []
                shifts = []
                for result in results:
                    if result is not None:
                        perms_solutions.append(result[0])
                        shifts.append(result[1])
            else:
                perms_solutions = []
                shifts = []
                for p in unique_perms:
                    B = permute_rows(self.Dis_matrix, p)
                    B = B[self.filter]

                    shift = find_shift(N, PINV, B)
                    if shift is not None:
                        perms_solutions.append(p)
                        shifts.append(shift)
                        
            return perms_solutions, shifts
        

    class GenericSumIntegral(DisMatrix):

        sectors = None
        vanishing_sectors = None
        permutations_to_canonical = None
        permutations_to_canonical_sign = None
        n_loops = None
        non_trivial_dens_positions = None

        def __init__(self):
            
            if self.__class__.sectors is None:
                self._initialize_class_cache()

        @classmethod
        def _initialize_class_cache(cls):

            cls.Dis = cls.Dis_matrix
            cls.n_loops = np.sum(np.sum(np.abs(cls.Dis) > 1e-5, axis=1) == 1)
            count_nonzero = np.sum(cls.Dis_matrix != 0, axis=1)
            cls.non_trivial_dens_positions = np.where(count_nonzero > 1)[0]

            cls.sectors = [format(i, f'0{len(cls.Dis)}b') for i in range(2**len(cls.Dis))]

            # Compute trivially vanishing sectors
            cls.vanishing_sectors = [s for s in cls.sectors if s.count('1') < cls.n_loops]

            for s in cls.sectors:
                if s not in cls.vanishing_sectors:
                    mask = [c == '1' for c in s]
                    if np.linalg.matrix_rank(cls.Dis[mask]) < cls.n_loops:
                        cls.vanishing_sectors.append(s)

            cls.sectors = [s for s in cls.sectors if s not in cls.vanishing_sectors]

            # Initialize dictionary of permutations to canonical sectors
            cls.permutations_to_canonical = {}
            cls.shifts_to_canonical = {}

            logging.info(" ")
            logging.info("Finding shift symmetries...")
            for i in range(cls.n_loops, len(cls.Dis)+1):
                
                perms, shifts = SumIntegralSuperSector(i).find_physical_sectors()
                cls.permutations_to_canonical.update(perms)

                # From shifts we want the leftover matrix, that is M @ X
                shifted_matrix = {}
                for sector, perms in shifts.items():
                    shifted_matrix_list = []
                    for perm, shift in perms.items():
                        el = cls.Dis @ shift
                        el[np.abs(el) < 1e-10] = 0

                        # Count number of 1 and -1 to save el or -el
                        count_1 = np.sum(np.isclose(el, 1))
                        count_m1 = np.sum(np.isclose(el, -1))
                        if count_m1 > count_1: el = -el
                        el_matrix = tuple(tuple(chop_to_int(i) for i in row) for row in el)
                        shifted_matrix_list.append([el_matrix, shift])
                        
                    shifted_matrix[sector] = dict(zip(perms, shifted_matrix_list))
                        
                cls.shifts_to_canonical.update(shifted_matrix)

            for vanishing_sector in cls.vanishing_sectors:
                cls.permutations_to_canonical.pop(vanishing_sector, None)
                cls.shifts_to_canonical.pop(vanishing_sector, None)

    # endregion


    # region Canonization    

    CANONIZATION_DICT = GenericSumIntegral().shifts_to_canonical
    non_trivial_dens_positions = GenericSumIntegral().non_trivial_dens_positions

    def is_separable(alpha):
        """ Dictates whether the sum-integral is separable in some of the variables and which variables """

        # Keep rows where there is a denominator
        masked = DisMatrix.Dis_matrix[np.array(alpha) != 0]

        nz_rows = np.count_nonzero(masked, axis=1)
        nz_cols = np.count_nonzero(masked, axis=0)
        nz_matrix = masked != 0

        condition = (
            nz_matrix &
            (nz_rows[:, None] == 1) &
            (nz_cols[None, :] == 1)
        )

        return np.argwhere(condition)[:,1]
    
    def choose_beta(betas1, betas2):
        """ Determine which list of betas is less complex, according first to
        their maximum power, then to their second maximum power, etc. If there is
        a tie, lastly, the shortest one is chosen.

        Returns 0 if betas1 == betas2, 1 if betas1 < betas2, 2 if betas2 < betas1.
        """

        sorted1 = sorted(list(set(beta_i for beta in betas1 for beta_i in beta if beta_i!=0)), reverse=True)
        sorted2 = sorted(list(set(beta_i for beta in betas2 for beta_i in beta if beta_i!=0)), reverse=True)
        if sorted1 == sorted2:
            if len(betas1) == len(betas2):
                return 0
            elif len(betas1) < len(betas2):
                return 1
            else:
                return 2
        elif sorted1 < sorted2:
            return 1
        else:
            return 2


    def shift_signature(sig: tuple, shifted_matrix: list):
        """ Finds loop momenta signatures after the momentum shift. """

        sig_np, shifted_matrix_np = np.array(sig), np.array(shifted_matrix)
        new_sig = np.linalg.solve(shifted_matrix_np[:len(sig)], sig_np)
        new_sig = tuple(chop_to_int(s) % 2 for s in new_sig)  
        return new_sig


    def expand_inv_props_internal(alpha, beta, sig):
        """ Expand sum-of-loop-momenta propagators with (-1) inverse powers if the result simplifies. """

        sol = {(alpha, beta, sig): 1}

        if not -1 in alpha:
            return sol

        minus_one_pos = [pos for pos, a in enumerate(alpha) if pos in non_trivial_dens_positions and a==-1]

        if not minus_one_pos:
            return sol
        
        dens = DisMatrix.Dis_matrix[minus_one_pos]

        for pos, den in zip(minus_one_pos, dens):
            nz = np.where(den != 0)[0]

            sol_new = {}
            for (ai, bi, si), coeff in sol.items():

                alpha_new = list(ai)
                alpha_new[pos] = 0
                alpha_new = tuple(alpha_new)
                separable_positions = is_separable(alpha_new)
                beta_check = [b + 1 if i in nz else b for i, b in enumerate(bi)]
                
                if any(beta_check[i] % 2 == 1 for i in separable_positions):

                    alpha1, alpha2 = list(alpha_new), list(alpha_new)
                    alpha1[nz[0]] -= 1
                    alpha2[nz[1]] -= 1

                    alpha1, alpha2 = tuple(alpha1), tuple(alpha2)

                    sol_new[(alpha1, bi, si)] = sol_new.get((alpha1, bi, si), 0) + coeff
                    sol_new[(alpha2, bi, si)] = sol_new.get((alpha2, bi, si), 0) + coeff

                else:
                    sol_new[(ai, bi, si)] = sol_new.get((ai, bi, si), 0) + coeff

            sol = sol_new

        sol_final = {}
        for sint, coeff in sol.items():
            for subsint, subcoeff in canonize_sint_internal(*sint).items():
                sol_final[subsint] = sol_final.get(subsint, 0) + coeff * subcoeff

        return sol_final     


    @cache
    def canonize_sint_internal(alpha, beta, sig):
        """ Canonizes a sum-integral given by (alpha, beta, sig) to the canonical form (alpha_new, beta_new, sig_new) 
        using the dictionary of shifts.
        """

        if len(beta) < len(alpha):
            fill_zero_beta = [0]*(len(alpha)-len(beta))
            beta = (*beta, *fill_zero_beta)

        # If the sum over zero modes is odd then this integral vanish
        if sum(beta) % 2 == 1:
            return {}

        # If it is separable in one variable and one of the zero modes is odd then this integral vanish
        separable_positions = is_separable(alpha)
        if any(beta[i] % 2 == 1 for i in separable_positions) and all(beta[i] == 0 for i in non_trivial_dens_positions):
            return {}

        # Identify to which sector this integral belongs to
        sector = get_sector(alpha)

        shifts_dict = CANONIZATION_DICT.get(sector, None)
        if shifts_dict is None:
            return {}

        alpha_new = None
        p_max = None

        # Find permutations that canonize propagators (alpha)
        for p in shifts_dict:
            permuted_alpha = [alpha[i] for i in p]  # apply permutation
            if (alpha_new is None) or (permuted_alpha > alpha_new):
                alpha_new = permuted_alpha
                p_max = p
                perms_list = [p_max]
            if (permuted_alpha == alpha_new):
                perms_list.append(p)

        alpha_new = tuple(alpha_new)

        # Find which of them make the canonized expression shortest and their corresponding betas
        shifted_matrix, _ = shifts_dict[perms_list[0]]
        beta_expansion = expand_fast(shifted_matrix, beta)
        candidates = [(beta_expansion, shifted_matrix)]

        # Choose canonization with smallest max(beta), and then shortest expansion
        for p in perms_list[1:]:
            shifted_matrix, _ = shifts_dict[p]
            beta_expansion = expand_fast(shifted_matrix, beta)

            match choose_beta(beta_expansion, candidates[0][0]):
                case 0:
                    candidates.append((beta_expansion, shifted_matrix))
                case 1:
                    candidates = [(beta_expansion, shifted_matrix)]
                case _:
                    pass
        
        # If any signature is non-zero (fermionic), prioritize the smallest or largest one (lexicographically)
        if any(sig):
            signatures = [shift_signature(sig, shifted_matrix) for _, shifted_matrix in candidates]
            if sig_order == "normal":
                chosen_pos = signatures.index(min(signatures))
            elif sig_order == "inverse":
                chosen_pos = signatures.index(max(signatures))
            else:                
                raise ValueError(f"sig_order must be 'normal' or 'inverse', but got {sig_order}")
            beta_new_list = candidates[chosen_pos][0]
            sig_new = signatures[chosen_pos]
        else:
            beta_new_list = candidates[0][0]
            sig_new = sig

        canonical = {(alpha_new, beta_new, sig_new) : chop_to_int(coeff) for beta_new, coeff in beta_new_list.items()}
        
        if len(canonical) > 1:

            fill_b = [0] * (len(beta) - len(list(beta_new_list)[0]))
            canonical = [(canonize_sint(a, (*b, *fill_b), s), coeff) for (a, b, s), coeff in canonical.items()]
            new_dict_canonicals = {}
            for dictionary, first_coeff in canonical:
                for sint, second_coeff in dictionary.items():
                    if sint not in new_dict_canonicals:
                        new_dict_canonicals[sint] = second_coeff * first_coeff
                    else:
                        new_dict_canonicals[sint] += second_coeff * first_coeff
        
            canonical = new_dict_canonicals

        elif len(canonical) == 1:
            # If it is separable in one variable and one of the zero modes is odd then this sum-integral vanishes
            separable_positions = is_separable(list(canonical)[0][0])
            if any(list(canonical)[0][1][i] % 2 == 1 for i in separable_positions):
                return {}

        return canonical
    
    @cache
    def canonize_sint(alpha, beta, sig):
        """ Canonizes a sum-integral and tries to expand inverse propagators. """

        flag = True
        sol = canonize_sint_internal(alpha, beta, sig)

        while flag:
            flag = False

            sol_new = {}
            for sint, coeff in sol.items():
                for subsint, subcoeff in expand_inv_props_internal(*sint).items():
                    sol_new[subsint] = sol_new.get(subsint, 0) + coeff * subcoeff

            if sorted(sol_new.items()) != sorted(sol.items()):
                flag = True
                sol = sol_new

        return sol
    
    # endregion
    

    def info_cache():
        print(expand_fast.cache_info())
    

    return CANONIZATION_DICT, canonize_sint, info_cache


### EXAMPLE OF USE ###
if __name__=="__main__":

    class DisMatrix3:
        Dis_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, -1, 0],
            [1, 0, -1],
            [0, 1, -1]
        ])

    CAN_DICT, can_func, _ = compute_shifts_dictionary(DisMatrix3)

    print(can_func((2, 2, 1, -1, 1, 1), (2, 2, 0), (0, 1, 0)))


