from sirena import generate_seeds, sirena, sints_to_txt
import random
from pathlib import Path
import time

bosonic_seeds = generate_seeds(loop_num=2,
                fer=False,
                max_r=8,
                max_s=4,
                alpha_ini=-2,
                sig_order='inverse',
                mass_dim={-6, -4, -2, 0, 2, 4, 6, 8},
                n_cpus='auto')

filtered_seeds = [sint for sint in bosonic_seeds if sint[0].count(0) == 0]
pos = random.sample(range(len(filtered_seeds)), 50)
bosonic_sample = [filtered_seeds[i] for i in pos]


fermionic_seeds = generate_seeds(loop_num=2,
                fer=True,
                max_r=8,
                max_s=4,
                alpha_ini=-2,
                sig_order='inverse',
                mass_dim={-6, -4, -2, 0, 2, 4, 6, 8},
                n_cpus='auto')

filtered_seeds = [sint for sint in fermionic_seeds if sint[0].count(0) == 0 and sint[2].count(1) > 0]
pos = random.sample(range(len(filtered_seeds)), 50)
fermionic_sample = [filtered_seeds[i] for i in pos]


script_dir = Path(__file__).parent

with open(script_dir / "2loop_sample.txt", "w") as f:
    f.write("sints_in = [\n")
    for sint in bosonic_sample:
        f.write(f"{sint}, \\\n")
    for sint in fermionic_sample:
        f.write(f"{sint}, \\\n")
    f.write("]")



with open(script_dir / "fermionic_sample.txt", "w") as f:
    f.write("sints_in = [\n")
    for sint in fermionic_sample:
        f.write(f"{sint}, \\\n")
    f.write("]")

