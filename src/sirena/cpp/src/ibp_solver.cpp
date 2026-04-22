#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

// Custom wrapper for FLINT polynomials
#include "FlintPoly.hpp"

/* ---------------------------------------------------------------------- */
/* Classes and functions to solve an IBP system with symbolic polynomials */
/* ---------------------------------------------------------------------- */

// C++ equivalent of dictionaries (key-value pairs) to store IBP equations
using Equation = std::map<int, FlintPoly>;
using Block = std::pair<int, std::vector<Equation>>;

// Groups system of IBP equations in blocks with common most complex sum-integral
std::vector<Block> ibps_to_blocks(const std::vector<Equation>& ibps) {
    // std::map orders keys in decreasing order automatically
    std::map<int, std::vector<Equation>, std::greater<int>> grouped_blocks;

    for (const auto& ibp : ibps) {
        if (ibp.empty()) continue;
        
        int min_tag = ibp.begin()->first; 
        grouped_blocks[min_tag].push_back(ibp);
    }

    // Convert map to block list
    std::vector<Block> result;
    for (const auto& pair : grouped_blocks) {
        result.push_back({pair.first, pair.second});
    }

    return result;
}

// Replace a sum-integral in an equation given its expression in terms of less complex sum-integrals
std::vector<Equation> replace_sint(int var, const std::vector<Equation>& eqs) {
    if (eqs.size() <= 1) return {};

    Equation eq_sub = eqs[0];
    std::vector<Equation> new_eqs;

    // Iterate over the rest of the equations
    for (size_t i = 1; i < eqs.size(); ++i) {
        Equation eq_target = eqs[i];

        // If variable not present, keep equation as is
        if (eq_target.find(var) == eq_target.end()) {
            new_eqs.push_back(eq_target);
            continue;
        }

        FlintPoly coeff = eq_target[var];
        eq_target.erase(var);

        FlintPoly c_var = eq_sub[var];

        // Substitute using eq_sub
        for (const auto& [v, c] : eq_sub) {
            if (v == var) continue;
            
            if (eq_target.find(v) == eq_target.end()) {
                eq_target[v] = (coeff * c) * FlintPoly(-1); // -coeff * c
            } else {
                eq_target[v] = (c_var * eq_target[v]) - (coeff * c);

                if (eq_target[v].is_zero()) {
                    eq_target.erase(v);
                }
            }
        }

        // Multiply remaining terms by c_var if not in eq_sub
        for (auto& [v, poly] : eq_target) {
            if (eq_sub.find(v) == eq_sub.end()) {
                poly = poly * c_var;
            }
        }

        // Simplify by GCD
        if (eq_target.size() > 1) {
            auto it = eq_target.begin();
            FlintPoly current_gcd = it->second;
            ++it;
            for (; it != eq_target.end(); ++it) {
                current_gcd = gcd(current_gcd, it->second);
            }

            for (auto& [v, poly] : eq_target) {
                poly = poly / current_gcd; 
            }
            new_eqs.push_back(eq_target);
        } 
        else if (eq_target.size() == 1) {
            Equation trivial_eq;
            trivial_eq[eq_target.begin()->first] = FlintPoly(1);
            new_eqs.push_back(trivial_eq);
        }
    }

    return new_eqs;
}

// Gauss forward elimination algorithm
std::vector<std::pair<int, Equation>> gauss_fwd_elim(std::vector<Block> blocked_ibps, bool verbose) {
    bool flag = false;
    int blocksize = blocked_ibps.size();
    
    while (!flag) {
        flag = true;
        std::vector<Equation> flat_ibps_new;
        int counter = 0;

        for (const auto& block_pair : blocked_ibps) {
            int mc_sint = block_pair.first;
            const auto& block = block_pair.second;

            if (block.size() > 1) {
                flag = false;
                counter++;

                flat_ibps_new.push_back(block[0]);
                auto replaced = replace_sint(mc_sint, block);
                flat_ibps_new.insert(flat_ibps_new.end(), replaced.begin(), replaced.end());
            } else {
                flat_ibps_new.insert(flat_ibps_new.end(), block.begin(), block.end());
            }
        }

        if (verbose) {
            std::cout << "\rBlocks left: " << counter << " / " << blocksize << "    " << std::flush;
        }
        
        blocked_ibps = ibps_to_blocks(flat_ibps_new); 
    }

    std::vector<std::pair<int, Equation>> flat_ibps;
    for (const auto& block_pair : blocked_ibps) {
        flat_ibps.push_back({block_pair.first, block_pair.second[0]});
    }

    if (verbose) {
        std:: cout << std::endl;
    }

    return flat_ibps;
}

// Back substitution algorithm
std::vector<std::pair<int, Equation>> gauss_bwd_elim(std::vector<std::pair<int, Equation>> fwd_ibps, bool verbose) {
    auto bwd_ibps = fwd_ibps;

    for (size_t i = 0; i < bwd_ibps.size(); ++i) {
        Equation eq = bwd_ibps[i].second;
        
        for (size_t k = 0; k < i; ++k) {
            int tag_to_replace = bwd_ibps[k].first;
            Equation eq_to_replace = bwd_ibps[k].second;
            
            std::vector<Equation> to_substitute = {eq_to_replace, eq};
            auto replaced = replace_sint(tag_to_replace, to_substitute);
            
            if (!replaced.empty()) {
                eq = replaced[0];
            }
        }

        bwd_ibps[i].second = eq;
        
        if (verbose) {
            std::cout << "\rRows solved: " << i+1 << " / " << bwd_ibps.size() << "    " <<std::flush;
        }

    }

    return bwd_ibps;
}


// Solve IBP system via Gauss forward elimination and back substitution
std::vector<std::pair<int, Equation>> solve_ibps(const std::vector<Equation>& ibps, bool verbose) {
    auto out = ibps_to_blocks(ibps);

    if (verbose) {
        std::cout << "Performing forward elimination on " << out.size() << " blocks..." << std::endl;
    }

    auto out2 = gauss_fwd_elim(out, verbose);

    if (verbose) {
        std::cout << "\nPerforming back substitution on " << out2.size() << " rows..." << std::endl;
    }

    auto out3 = gauss_bwd_elim(out2, verbose);

    return out3;
}


/* ------------------------------ */
/* Binding between C++ and Python */
/* ------------------------------ */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

namespace py = pybind11;

// ------------ Module -------------
PYBIND11_MODULE(ibp_solver, m) {
    m.doc() = "Symbolic IBP solver";

    // Class FlintPoly for python
    py::class_<FlintPoly>(m, "FlintPoly")
        .def(py::init<>())       
        .def(py::init<int>())    
        .def(py::init<std::vector<long>>())
        .def("__repr__", &FlintPoly::to_string)
        .def("__str__", &FlintPoly::to_string)
        .def("get_coeffs", &FlintPoly::get_coeffs, "Returns the coefficients of the polynomial as a list, where the index is the exponent")
        ;

    // Export functions to Python
    m.def("ibps_to_blocks", &ibps_to_blocks, "Groups IBP equations");
    m.def("replace_sint", &replace_sint, "Replaces variables in the system");
    m.def("gauss_fwd_elim", &gauss_fwd_elim, py::arg("blocked_ibps"), py::arg("verbose") = true, "Forward elimination of Gauss");
    m.def("gauss_bwd_elim", &gauss_bwd_elim, py::arg("fwd_ibps"), py::arg("verbose") = true, "Back substitution of Gauss");
    m.def("solve_ibps", &solve_ibps, py::arg("ibps"), py::arg("verbose") = true, "Performs the entire process of solving IBP equations");
}

//