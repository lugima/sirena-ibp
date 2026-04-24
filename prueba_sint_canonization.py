import itertools
import time

def generate_betas(length, target_sum):
    """Genera tuplas exactas para beta usando Stars and Bars (elementos >= 0)."""
    if target_sum < 0: return
    if length == 1:
        yield (target_sum,)
        return
    for c in itertools.combinations(range(target_sum + length - 1), length - 1):
        yield tuple(b - a - 1 for a, b in zip((-1,) + c, c + (target_sum + length - 1,)))

def generate_alphas(length, target_sum, alpha_ini, max_val, neg_allowance):
    """
    Generador recursivo ultra-rápido para alpha.
    'neg_allowance' controla que la suma de negativos nunca baje de alpha_ini.
    Corta instantáneamente las ramas matemáticas que no pueden llegar al 'target_sum'.
    """
    if length == 0:
        if target_sum == 0:
            yield ()
        return
        
    for val in range(alpha_ini, max_val + 1):
        # Actualizamos la tolerancia de números negativos
        new_allowance = neg_allowance + val if val < 0 else neg_allowance
        
        if new_allowance >= 0:
            rem_len = length - 1
            rem_target = target_sum - val
            
            # Poda (Pruning): ¿Es matemáticamente posible llegar al target con lo que queda?
            min_possible = -new_allowance
            max_possible = rem_len * max_val
            
            if min_possible <= rem_target <= max_possible:
                for tail in generate_alphas(rem_len, rem_target, alpha_ini, max_val, new_allowance):
                    yield (val,) + tail


# --- TUS VARIABLES (Ajusta según necesites) ---
loop_num = 3
alpha_ini = -2  # Soporta números negativos perfectamente
max_r = 6
max_s = 6
fer = True       # o False
mass_dim = {2}   # Set de dimensiones permitidas

# --- LÓGICA OPTIMIZADA ---
ini = time.perf_counter()
den_combinations = loop_num * (loop_num + 1) // 2

# Signatures
bf_list = list(itertools.product(range(1), repeat=loop_num)) if fer else [(0,) * loop_num]

# Límite superior individual para un elemento de alpha basado en tu lógica original
max_alpha_val = max_r - alpha_ini 

filtered_seeds_list = []

for dim in mass_dim:
    # Constante para esta dimensión: 4*L - dim
    K = 4 * loop_num - dim
    
    # sum(beta) solo puede ir de 0 a max_s
    for s_beta in range(0, max_s + 1):
        
        # Despejamos la suma requerida para alpha: 2 * sum(alpha) = K + sum(beta)
        s_alpha_2 = K + s_beta
        
        # Si no es divisible por 2, es matemáticamente imposible que sum(alpha) sea entera
        if s_alpha_2 % 2 == 0:
            s_alpha = s_alpha_2 // 2
            
            # Comprobamos que sum(alpha) está dentro de tus límites teóricos (<= max_r)
            if alpha_ini <= s_alpha <= max_r:
                
                # Generamos SOLO las tuplas exactas
                valid_betas = list(generate_betas(den_combinations, s_beta))
                allowance = -alpha_ini if alpha_ini < 0 else 0
                valid_alphas = list(generate_alphas(den_combinations, s_alpha, alpha_ini, max_alpha_val, allowance))
                
                # Si existen tuplas válidas en ambos lados, hacemos el producto cartesiano
                # solo de estas listas pequeñitas y válidas
                if valid_alphas and valid_betas:
                    for bf in bf_list:
                        for a in valid_alphas:
                            for b in valid_betas:
                                filtered_seeds_list.append((a, b, bf))

# Resultado final empaquetado
filtered_seeds = tuple(filtered_seeds_list)
fin = time.perf_counter()
print(f"Semillas generadas al instante: {len(filtered_seeds)}")
print(f"Tiempo de ejecución: {fin - ini:.2f} segundos")



# --- LÓGICA ORIGINAL ---
# Signatures
ini = time.perf_counter()
bf_list = list(itertools.product(range(1), repeat=loop_num))

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

fin = time.perf_counter()
print(f"Semillas generadas al instante: {len(filtered_seeds)}")
print(f"Tiempo de ejecución: {fin - ini:.2f} segundos")
