import argparse
import time
import sys
import logging
from importlib import resources

from .in_out_utils import get_max_r_s, sints_from_txt, params_from_txt, sints_to_txt
from .reduction import sirena

def args_main():

    if '--demo' in sys.argv:
        logging.basicConfig(level=logging.DETAILED_INFO, format='%(message)s')
        logging.info("\nRunning demo example...\n\n")
        run_demo()
        logging.info("\n\nDemo finished successfully!\n")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Sum-Integral REductioN Algorithm (SIRENA)")
    
    parser.add_argument("input", metavar="INPUT", help="input file with sum-integrals")
    parser.add_argument("output", metavar="OUTPUT", help="output file with reductions")
    parser.add_argument("-p", "--params", type=str, default="None", metavar="PARAMS", help="parameters file")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="enable printing reduction progress: -v (info) or -vv (detailed info)")
    parser.add_argument("--demo", action='store_true', help="runs a quick example to check the installation")

    args = parser.parse_args()

    return args


def run_demo():

    with resources.path("sirena.examples", "fermion_test.txt") as path:

        input_dir = path
        sints_in, coeffs_in, priority = sints_from_txt(input_dir)

        sints_in = sints_in[:4]

        logging.info("Reducing the following 2-loop fermionic sum-integrals:\n")
        for sint in sints_in:
            logging.info(f"{sint}\n")

        time.sleep(1.5)
        sols = sirena(sints_in, basis_sints=priority)


def main():

    args = args_main()

    input_path, output_path, params_path = args.input, args.output, args.params
    verbose = args.verbose

    match verbose:
        case 0:
            logging.basicConfig(level=logging.WARNING, format='%(message)s')
        case 1:
            logging.basicConfig(level=logging.INFO, format='%(message)s')
        case _:
            logging.basicConfig(level=logging.DETAILED_INFO, format='%(message)s')
   
    sints_in, coeffs_in, priority = sints_from_txt(input_path)
    params = params_from_txt(params_path)

    # Detect if it's a single entry: (dens, nums, sig)
    if len(sints_in) == 3 and len(sints_in[0]) != len(sints_in[2]):
        sints_in = (sints_in,)  # wrap into a tuple of one element

    max_r_in, max_s_in = get_max_r_s(sints_in)
    if params["max_r"] < max_r_in:
        logging.error(f"The maximum propagator power for seed generation in the parameters file ({params["max_r"]}) " +
                      f"must be equal to or larger than that of the input sum-integrals ({max_r_in}).\n" +
                      f"Defaulting to max_r = {max_r_in}")
        params["max_r"] = max_r_in
    if params["max_s"] < max_s_in:
        logging.error(f"The maximum numerator power for seed generation in the parameters file ({params["max_s"]}) " +
                      f"must be equal to or larger than that of the input sum-integrals ({max_s_in}).\n" +
                      f"Defaulting to max_s = {max_s_in}")
        params["max_s"] = max_s_in

    # Reduce IBP system
    t_ini = time.time()
    sols = sirena(sints_in, max_r=params["max_r"], max_s=params["max_s"], alpha_ini=params["alpha_ini"],
                  sig_order=params["sig_order"], n_cpus=params["n_cpus"], rerun=params["rerun"],
                  basis_sints=priority)
    t_fin = time.time()

    print(f"\nFinished in {t_fin-t_ini:.2f} seconds\n")

    # Output
    sints_to_txt(sints_in, sols, coeffs_in=coeffs_in, file=output_path, to_wolfram=params["to_wolfram"])

if __name__ == "__main__":

    main()