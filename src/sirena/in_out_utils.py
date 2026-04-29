import logging
import sys
import time

import numpy as np
import sympy as sp

# region Read/write

def sints_from_txt(file: str):
    """ Loads sints_in (required), and optional coeffs_in and priority from a .txt file. """

    with open(file, 'r') as f:
        text = f.read()

    text = text.replace("\\\n", "").replace("\n", "").replace(" ", "")

    # Split into key=value chunks
    parts = [p for p in text.split("sints_in=") if p]

    data = {}

    # Rebuild full structure safely
    text = "sints_in=" + parts[0] if parts else ""

    for key in ["sints_in", "coeffs_in", "priority"]:
        if f"{key}=" in text:
            # Split only once after the key
            after_key = text.split(f"{key}=", 1)[1]

            # Find where this value ends (next key or end of string)
            next_positions = [
                after_key.find(k + "=") for k in ["sints_in", "coeffs_in", "priority"]
                if k + "=" in after_key
            ]
            next_positions = [pos for pos in next_positions if pos != -1]

            end = min(next_positions) if next_positions else len(after_key)
            value = after_key[:end]

            data[key] = value

    # Parse values
    if "sints_in" not in data:
        raise ValueError("A list of sum-integrals sints_in was not found in the input file.")
    sints_in = eval(data["sints_in"])

    coeffs_in = None
    if "coeffs_in" in data:
        coeffs_in = sp.sympify(data["coeffs_in"])

    priority = []
    if "priority" in data:
        priority = eval(data["priority"])

    return sints_in, coeffs_in, priority


def params_from_txt(file=None):
    """ Load reduction parameters from a .txt file as dictionary. """

    if file == None or file == "None":
        # Default parameters
        params = {
            "max_r": 6, 
            "max_s": 6, 
            "alpha_ini": 0, 
            "sig_order": "normal",
            "rerun": True, 
            "n_cpus": ("auto", 1), 
            "to_wolfram": False}
    else:
        params = {}
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines or comments
                if not line or line.startswith("#"):
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                params[key] = eval(value)

        # Exit if params file format is incorrect
        if (
            type(params["max_r"]) != int or 
            type(params["max_s"]) != int or
            type(params["alpha_ini"]) != int or
            type(params["sig_order"]) != str or
            type(params["rerun"]) != bool or
            not isinstance(params["n_cpus"], (list, tuple)) or
            type(params["to_wolfram"]) != bool
        ):
            logging.error("Invalid argument types in parameter file.")
            logging.error("\nExpected format:\n\n" +
                          "\tmax_r : int\n" +
                          "\tmax_s : int\n" +
                          "\talpha_ini : int\n" +
                          "\tsig_order : str\n" +
                          "\trerun : bool\n" +
                          "\tn_cpus = (int|\"auto\", int|\"auto\")\n" +
                          "\tto_wolfram = bool\n")

            logging.error("Aborting...")
            sys.exit(0)

    return params


def sints_to_txt(sints_in: list, sols: list, file: str, coeffs_in=None, to_wolfram=False):
    """ Writes reduction to .txt file.

    If to_wolfram=True, output is parsed as Wolfram Language
    """

    def combine_eqs(coeffs, eqs):
        """ Combines the solutions for a linear combination of sum-integrals that the IBP system has been solved for. """

        result = {}
        for c, eq in zip(coeffs, eqs):
            if eq:
                for sint in eq:
                    result[sint] = result.get(sint, 0) + c * eq[sint]

        result = {tag: sp.factor(pol) for tag, pol in result.items() if pol != 0}

        return {k: v for k, v in result.items() if v!=0}


    def sandwich(expr: str):
        return "{" + expr + "}"


    def sint_to_string(alpha: tuple, beta: tuple, sig: tuple) -> str:
        """ Converts sum-integral index list to string. """

        if not to_wolfram:
            alpha_string = "".join(str(a) for a in alpha)
            beta_string = "".join(str(b) for b in beta)
            sig_string = "".join(str(s) for s in sig)

            return f"i_{alpha_string};{beta_string};{sig_string}"
        else:
            alpha_string = sandwich(",".join(str(a) for a in alpha))
            beta_string = sandwich(",".join(str(b) for b in beta))
            sig_string = sandwich(",".join(str(s) for s in sig))

            return f"i[{alpha_string},{beta_string},{sig_string}]"


    def format_term(sint, coeff):
        """ Formats sum-integral - coefficient pairs as string. """
        
        sint_str = sint_to_string(*sint)
        coeff_str = str(coeff)
        if to_wolfram:
            coeff_str = str(coeff_str).replace("**", "^").replace("_", "")

        if coeff_str != "1":
            return f"({coeff_str}) * {sint_str}"
        else:
            return f"{sint_str}"


    masters = sorted({format_term(sint, "1") 
                            for sol in sols for sint in sol})
    masters_str = ", ".join(masters)

    with open(file, "w", encoding="utf-8") as f:

        if to_wolfram:    
            f.write("(*\n")
            f.write(" This file has been automatically generated by SIRENA \n")
            f.write("*)\n\n")

            f.write("(*\n")
            f.write("=" * 60 + "\n")
            f.write(" LIST OF REDUCTIONS\n")
            f.write("=" * 60 + "\n")
            f.write("*)\n\n")
        else:
            f.write("This file has been automatically generated by SIRENA \n\n")

            f.write("=" * 60 + "\n")
            f.write(" LIST OF REDUCTIONS\n")
            f.write("=" * 60 + "\n\n")

        last = len(sints_in) - 1

        if to_wolfram:
            
            # Write as Mathematica replacement rules
            f.write("sintsOut = {\n\n")

            for i, (sint_in, sol) in enumerate(zip(sints_in, sols)):
                lhs = sint_to_string(*sint_in)
                terms = [format_term(sint, coeff) for sint, coeff in sol.items()]
                rhs = " + ".join(terms)

                f.write(f"{lhs} ->\n")
                if rhs != "":
                    if i != last:
                        f.write(f"{rhs},\n\n")
                    else:
                        f.write(f"{rhs}\n\n")
                else:
                    # Vanishing results
                    if i != last:
                        f.write("0,\n\n") 
                    else:
                        f.write("0\n\n")
            
            f.write("}\n\n")

            # Masters
            f.write("masters = {\n")
            f.write(masters_str)
            f.write("\n}\n\n")
        
        else:

            # Write as equalities
            f.write("sints_out = [\n\n")

            for i, (sint_in, sol) in enumerate(zip(sints_in, sols)):
                lhs = sint_to_string(*sint_in)
                terms = [format_term(sint, coeff) for sint, coeff in sol.items()]
                rhs = " + ".join(terms)
                
                f.write(f"{lhs} =\n")
                if rhs != "":
                    if i != last:
                        f.write(f"{rhs},\n\n")
                    else:
                        f.write(f"{rhs}\n\n")
                else:
                    # Vanishing results
                    if i != last:
                        f.write("0,\n\n") 
                    else:
                        f.write("0\n\n")

            f.write("]\n\n")

            # Masters
            f.write("masters = [\n")
            f.write(masters_str)
            f.write("\n]\n\n")

        # Final combination
        if coeffs_in is not None:
            final = combine_eqs(coeffs_in, sols)

            if to_wolfram:
                f.write("(*\n")
                f.write("=" * 60 + "\n")
                f.write(" COMBINED REDUCTION\n")
                f.write("=" * 60 + "\n")
                f.write("*)\n\n")
            else:
                f.write("=" * 60 + "\n")
                f.write(" COMBINED REDUCTION\n")
                f.write("=" * 60 + "\n\n")

            terms = [
                format_term(m_sint, coeff)
                for m_sint, coeff in final.items()
            ]

            f.write("combined =\n")

            for i, term in enumerate(terms):
                if i == 0:
                    f.write(f"{term} ")
                else:
                    f.write(f"+ {term}")

            f.write("\n")

    print(f"Output successfully written to '{file}'")


# endregion


# region Preparing input

def propagator_matrix(L: int):
    """ Generates a propagator matrix with pairwise differences at given loop order. """

    if L <= 1:
        raise ValueError(f"Loop order must be greater than 1. {L} was provided")

    rows = []

    # Identity part: P, Q, R, ...
    for i in range(L):
        vec = [0] * L
        vec[i] = 1
        rows.append(vec)

    # Pairwise differences: Pi - Pj
    for i in range(L):
        for j in range(i + 1, L):
            vec = [0] * L
            vec[i] = 1
            vec[j] = -1
            rows.append(vec)

    return np.array(rows)


def are_fermionic(sints: tuple) -> bool:
    """ Returns a boolean indicating whether there are fermionic signatures in a list of sum-integrals. """

    signatures = [sig for sint in sints for sig in sint[2]]

    return 1 in signatures


def get_mass_dim(sints: tuple):
    """ Returns the mass dimensions of a list of sum-integrals. """

    loop_order = len(sints[0][2])

    mass_dims = set([4 * loop_order - 2 * sum(list(sint[0])) + sum(list(sint[1]))
                 for sint in sints])
    
    return mass_dims

def get_most_neg_alpha(sints: tuple) -> int:
    """ Returns the most negative denominator power of a list of sum-integrals. """

    most_neg_alpha = min([alpha for sint in sints for alpha in sint[0]])

    if most_neg_alpha > -1:
        return 0
    else:
        return most_neg_alpha
    
def get_max_r_s(sints: tuple) -> bool:
    """ Returns the maximum propagator and numerator powers of a list of sum-integrals. """

    # Split denominator and numerator indices. Also unpack (and ignore) signature
    dens, nums, _ = map(list, zip(*sints))

    max_r_in = max([sum(i for i in den) for den in dens])
    max_s_in = max([sum(i for i in num) for num in nums])

    return max_r_in, max_s_in
    

def print_cafe():
    cafe = r"""
        
          ( (
           ) )
        ........
        |      |]
        \      /    You should go get a coffee while SIRENA runs, it might take a while!
         `----'
    
    
    """
    print(cafe)
    time.sleep(3)