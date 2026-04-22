cmake_minimum_required(VERSION 3.15...3.27)
project(cpp_modules LANGUAGES CXX)

# Find pybind11 (scikit-build-core provides it)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find system libreries (FLINT y GMP)
find_library(FLINT_LIB NAMES flint REQUIRED)
find_library(GMP_LIB NAMES gmp REQUIRED)

find_path(FLINT_INCLUDE_DIR 
          NAMES flint/flint.h 
          PATHS 
          /usr/local/include 
          /opt/homebrew/include
          /usr/include
          REQUIRED
)

# ==========================================
# 1. Module: ibp_solver
# ==========================================
pybind11_add_module(ibp_solver "src/ibp_solver.cpp")

# Equivalent to -std=c++14 -O3 -Wall -lflint -lgmp
target_compile_features(ibp_solver PRIVATE cxx_std_14)
target_compile_options(ibp_solver PRIVATE -Wall -O3)
target_link_libraries(ibp_solver PRIVATE ${FLINT_LIB} ${GMP_LIB})

target_include_directories(ibp_solver PRIVATE 
    "includes"
    ${FLINT_INCLUDE_DIR}
)

# ==========================================
# 2. Module: ibp_solver_num
# ==========================================
pybind11_add_module(ibp_solver_num src/ibp_solver_num.cpp)

# Equivalent to -std=c++14 -O3 -Wall -lgmp
target_compile_features(ibp_solver_num PRIVATE cxx_std_14)
target_compile_options(ibp_solver_num PRIVATE -Wall -O3)
target_link_libraries(ibp_solver_num PRIVATE ${GMP_LIB})

target_include_directories(ibp_solver_num PRIVATE 
    "includes"
)


# ==========================================
# 3. Installation 
# ==========================================
install(TARGETS ibp_solver ibp_solver_num DESTINATION sirena)