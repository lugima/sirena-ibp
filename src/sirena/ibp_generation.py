import math
import numpy as np
import sympy as sp

def generate_exponents(n: int, k: int, prefix: list = []):
    """ Generates all non-negative tuples (k1,...,kk) that sum n """

    if k == 1:
        yield prefix + [n]
    else:
        for i in range(n + 1):
            yield from generate_exponents(n - i, k - 1, prefix + [i])


def pre_build_ibp_relations(n_loop: int, Dis: np.ndarray):
    """ Builds the generic (symbolic) IBP relations for a given loop order and list of denominators, 
    and returns a function to generate them for any sum-integral as a seed 
    """

    # Dis is the matrix of denominators
    # When a vector is represented as a single integer,
    # it refers to its position within this list.
    simple_den = np.where(np.count_nonzero(Dis, axis=1) == 1)[0]
    composed_den = np.where(np.count_nonzero(Dis, axis=1) > 1)[0]

    if n_loop != len(simple_den):
        raise ValueError("Number of loops don't coincide with number of rows with just a single 1.")

    n = len(Dis)
    alpha = sp.symbols(f'a0:{n}')
    beta = sp.symbols(f'b0:{n}')
    d = sp.symbols('d')


    class LadderOperator:
        """ Class to manage raising and lowering arguments of SumIntegrals.
        The first argument is a dictionary representing which position
        of the alphas should be raised or lowered and by how much
        The second is equivalent but for the betas
        """
        def __init__(self, alphas = {}, betas = {}):
            self.dict_alphas = alphas
            self.dict_betas = betas


        def reduce_ladder(self):
            """ Reduces the LadderOperator to a sum of LadderOperators that can be applied directly to the SumIntegrals, 
            i.e., with only positive powers of the simple denominators, by applying multionimial expansion for the composed denominators
            """

            ladder_pre = {pos : value for pos, value in self.dict_betas.items() if pos in simple_den}
            result_ladder = [ladder_pre]
            result_coeff = [1]
            for pos, value in self.dict_betas.items():
                if pos in composed_den:
                    
                    monomials = []
                    coefficients = []
                    for exponents in generate_exponents(value, n_loop):
                        # Multinomial coefficient: value! / (k1! * ... * kn!)
                        coef = math.factorial(value)
                        for e in exponents:
                            coef //= math.factorial(e)
                        for a, e in zip(Dis[pos], exponents):
                            coef *= a**e
                        if coef != 0:
                            monomials.append(tuple(exponents))
                            coefficients.append(int(coef))

                    # List of dictionaries representing the different Ladder operators
                    monomials = [
                        {pos : val for pos, val in enumerate(m) if val!=0}
                        for m in monomials
                    ]

                    result_ladder_new = []
                    result_coeff_new = []
                    for ladd_pre, coeff_pre in zip(result_ladder, result_coeff):
                        for coeff, mon in zip(coefficients, monomials):
                            new_ladd = ladd_pre.copy()
                            for pos, val in mon.items():
                                new_ladd[pos] = new_ladd.get(pos, 0) + val
                            result_ladder_new.append(new_ladd)
                            result_coeff_new.append(coeff_pre * coeff)

                    result_ladder = result_ladder_new
                    result_coeff = result_coeff_new

            result_ladder = [LadderOperator(self.dict_alphas, b) for b in result_ladder]
            total = Sum(*zip(result_coeff, result_ladder))

            return total
                    

        def __eq__(self, other):
            if isinstance(other, LadderOperator):
                return self.dict_alphas == other.dict_alphas and self.dict_betas == other.dict_betas
            return NotImplemented

        def __repr__(self):
            return f"LadderOperator({self.dict_alphas};{self.dict_betas})"
        

    class SumInt:
        """ Class to store SumIntegrals.
        alpha -> Vector of powers of denominators
        beta -> vector of powers of zero modes

        I(alpha, beta) = (P0_i ^ beta_i) / (P_i^2) ^ alpha_i
        """
        def __init__(self, alpha, beta):
            self.alpha = alpha
            self.beta = beta

        def apply_ladder(self, ladder : LadderOperator):
            """Apply a LadderOperator to this SumIntegral """

            alpha_new = list(self.alpha)
            for k, v in ladder.dict_alphas.items():
                alpha_new[k] += v
            
            beta_new = list(self.beta)
            for k, v in ladder.dict_betas.items():
                beta_new[k] += v

            return SumInt(tuple(alpha_new), tuple(beta_new))

        def __repr__(self):
            return f"I({self.alpha};{self.beta})"
            

    class Sum:
        """ Class to represent sums of LadderOperators with coefficients
        Terms are arranged in a list [(coeff1, LadderOperator1), ...]
        """

        def __init__(self, *args):
            args_with_coeffs = [arg for arg in args if isinstance(arg, tuple)]
            
            self.terms = [(1, arg) for arg in args if not isinstance(arg, tuple)]
            self.terms += [(coeff, term) for coeff, term in args_with_coeffs if coeff!=0]

        def __mul__(self, factor):
            if isinstance(factor, int) or isinstance(factor, float):
                return Sum(*[(coeff * factor, term) for coeff, term in self.terms])
            
        def __rmul__(self, factor):
            return self * factor
        
        def __add__(self, other):
            # Try to combine coefficients of same LadderOperator
            if isinstance(other, Sum):
                repeated_terms = [(coeff1+coeff2, term1) for coeff1, term1 in self.terms
                                for coeff2, term2 in other.terms if term1==term2]
                terms1 = [term for _, term in self.terms]
                terms2 = [term for _, term in other.terms]
                non_repeated_terms1 = [(coeff, term) for coeff, term in self.terms if term not in terms2]
                non_repeated_terms2 = [(coeff, term) for coeff, term in other.terms if term not in terms1]
                return Sum(*repeated_terms, *non_repeated_terms1, *non_repeated_terms2)

        def __radd__(self, other):
            return self + other
        
        def __neg__(self):
            return Sum(*[(-coeff, term) for coeff, term in self.terms])
        
        def __sub__(self, other):
            return self + (-other) 
        
        def __floordiv__(self, other):
            if isinstance(other, int):
                return Sum(*[(coeff // other, term) for coeff, term in self.terms])

        def __repr__(self):
            mult = [f"({coeff}) * {term}" if coeff!=1 else f"{term}"
                    for coeff, term in self.terms]
            return " + ".join(mult)
        
        def __str__(self):
            mult = [f"({coeff}) * {term}" if coeff!=1 else f"{term}"
                    for coeff, term in self.terms]
            return " + ".join(mult)


    def contraction(i : int, j : int) -> Sum:
        """ Compute the contraction of the vectors Dis[i] · Dis[j].
        It gives the result as a Sum with coefficients multiplying LadderOperators

        The output is multiplied by 2, to avoid 1/2 factors in the process
        """
        
        # We have pi^2 = P^2 - P0^2
        if i == j:
            return 2 * Sum(LadderOperator(alphas = {i : -1})) - 2 * LadderOperator(betas = {i : 2}).reduce_ladder()
        
        # Try to express p·q as [-(p-q)^2 + p^2 + q^2] / 2
        # or as [(p+q)^2 - p^2 - q^2] / 2,
        # according to whether p+q or p-q is in Dis.
        if i in simple_den and j in simple_den:

            for z, vec in enumerate(Dis):
                if np.array_equal(Dis[i] + Dis[j], vec):
                    return (contraction(z, z) - contraction(i, i) - contraction(j, j)) // 2
                elif np.array_equal(Dis[i] - Dis[j], vec) or np.array_equal(Dis[j] - Dis[i], vec):
                    return (-contraction(z, z) + contraction(i, i) + contraction(j, j)) // 2

            return NotImplementedError             

        # If the first vector is a combination of other two, e.g. p+q,
        # then exploits linearity in the first argument
        if i in composed_den:
            list_terms = [coeff * contraction(ii, j) for ii, coeff in enumerate(Dis[i]) if coeff != 0]
            result = list_terms[0]
            for term in list_terms[1:]:
                result += term
            return result
        
        # Exploit symmetry (and thus, linearity in the second argument)
        return contraction(j,i)
        
        
    def derivative(vec : int, sint : SumInt, vec_diff : int) -> dict:
        """ Computes the contraction of vec with the derivative of the
        sumint wrt vec_diff. Namely,

            A = vec_mu d(sumint)/d(vec_diff_mu) 

        Returns a dictionary with each SumIntegral as key (represented directly
        by its arguments as a tuple) and the coefficient as value.
        
        We use the formula
            A = sum_i d(sumint)/d(Di^2) 2 (Di_mu · vec_diff_mu) d(Di)/d(vec_diff)
        """

        alpha, beta = sint.alpha, sint.beta
        result = {}

        # Iterate through every denominator
        # i -> denominator
        # a -> power in sint
        for i, a in enumerate(alpha):

            # This corresponds to the most inside derivative,
            # i.e., the derivative of the vector of the denominator
            coeff_in_vec = Dis[i, vec_diff]

            if coeff_in_vec == 0:
                continue
            
            # Multiply with the power of the denominator,
            # coming from the derivative of 1/(Di^(2a))
            prefact = - a * coeff_in_vec
            
            # Compute the contraction of the denominator and the vector
            # This comes with the needed extra factor of 2
            contracted_vecs = contraction(i, vec)

            # Sum 1 to alpha_i
            # coming from the derivative of 1/(Di^(2a))
            alpha_new = list(alpha)
            alpha_new[i] += 1
            inew = SumInt(tuple(alpha_new), beta)

            # Apply all ladder operators coming from the contraction
            for coeff, ladder in contracted_vecs.terms:
                
                new_coeff = prefact * coeff
                new_sumint = inew.apply_ladder(ladder)

                args = (new_sumint.alpha, new_sumint.beta)

                if args in result:
                    result[args] += new_coeff
                else:
                    result[args] = new_coeff      

        return result


    def build_ibp(vec : int, sint : SumInt, vec_diff : int) -> dict:
        """ Returns the IBP relation coming from differentiating the product vec * sumint wrt vec_diff.
        The output is a dictionary representing a sum of SumInts and the coefficients.

        Computes the derivative of sumint wrt vec_diff and adds the extra factor
        of d if vec is equal to vec_diff
        """

        if vec not in simple_den or vec_diff not in simple_den:
            return NotImplemented

        result = derivative(vec, sint, vec_diff)
        
        if vec == vec_diff:
            first_sumint = (sint.alpha, sint.beta)
            if first_sumint in result:
                result[first_sumint] += d
            else:
                result[first_sumint] = d

        return result


    ibp_func_list = []
    for vec in range(n_loop):
        ibp_func_list.append([])
        for vec_diff in range(n_loop):
            ibp = build_ibp(vec, SumInt(alpha, beta), vec_diff)
            ibp_func_list[-1].append(sp.lambdify((alpha, beta, d), ibp, "math"))


    def build_ibp_lambdified(dmom_num: int, mom_num: int , sint: tuple, d=None) -> dict:
        """ Generates IBP identity for a given sum-integral (keys use tuples) """

        if not d:
            d = sp.Symbol('d')  
        result = ibp_func_list[mom_num][dmom_num](*sint, d)
        return result   

    return build_ibp_lambdified





if __name__=='__main__':

    Dis = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, -1, 0],
        [1, 0, -1],
        [0, 1, -1]
    ])
    n_loop = 3

    ibp_relations = pre_build_ibp_relations(n_loop, Dis)

    Dis = np.array([
        [1, 0],
        [0, 1],
        [1, -1]
    ])
    n_loop = 2

    ibp_relations = pre_build_ibp_relations(n_loop, Dis)

    a,b,c,A,B,C = sp.symbols('a b c A B C')
    print(ibp_relations(0,1,((a,b,c),(A,B,C))))


