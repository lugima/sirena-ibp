# Placeholder for the canonization function, to be imported from canonization_integrals_fer.py
canonize_sint = None
build_ibp = None 

def set_canonize_function(func):
    global canonize_sint
    canonize_sint = func

def set_build_ibp_function(func):
    global build_ibp
    build_ibp = func

def set_precomputed_functions(canonize_func = None, build_ibp_func = None):
    if canonize_func is not None:
        set_canonize_function(canonize_func)
    if build_ibp_func is not None:
        set_build_ibp_function(build_ibp_func)
