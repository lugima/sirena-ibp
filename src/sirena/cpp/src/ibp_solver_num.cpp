#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

/* ---------------------------------------------------------------- */
/* Classes and functions to solve an IBP system using finite fields */
/* ---------------------------------------------------------------- */

constexpr int MOD = 1000003; // prime modulus

struct Coeff {
    int v;

    // Constructor
    // Value is always in range 0 <= v <= MOD
    Coeff(int x = 0) : v((x % MOD + MOD) % MOD) {}

    // Operators
    Coeff operator+(const Coeff& other) const {
        return Coeff(v + other.v);
    }

    Coeff operator-(const Coeff& other) const {
        return Coeff(v - other.v);
    }

    Coeff operator*(const Coeff& other) const {
        return Coeff(1LL * v * other.v % MOD);
    }

    // Fermat modular inverse (only for prime MOD)
    static int modinv(int a) {
        int res = 1, p = MOD - 2;
        int base = a;
        // Fast exponentiation
        while (p) {
            if (p & 1) res = 1LL * res * base % MOD;
            base = 1LL * base * base % MOD;
            p >>= 1;
        }
        return res;
    }

    Coeff operator/(const Coeff& other) const {
        return Coeff(1LL * v * modinv(other.v) % MOD);
    }

    bool is_zero() const {
        return v == 0;
    }
};

// C++ equivalent of dictionaries (key-value pairs) to store IBP equations
using Equation = std::map<int, Coeff>;
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

        Coeff coeff = eq_target[var];
        eq_target.erase(var);

        Coeff c_var = eq_sub.at(var);

        // Substitute using eq_sub
        for (const auto& [v, c] : eq_sub) {
            if (v == var) continue;

            if (eq_target.find(v) == eq_target.end()) {
                eq_target[v] = Coeff(0) - (coeff * c); // -coeff * c
            } else {
                eq_target[v] = (c_var * eq_target[v]) - (coeff * c);

                if (eq_target[v].is_zero()) {
                    eq_target.erase(v);
                }
            }
        }

        // Multiply remaining terms by c_var if not in eq_sub
        for (auto& [v, val] : eq_target) {
            if (eq_sub.find(v) == eq_sub.end()) {
                val = val * c_var;
            }
        }

        // Normalization (finite field)
        if (!eq_target.empty()) {
            Coeff lead = eq_target.begin()->second;

            // Divide all coefficients by leading coefficient
            for (auto& [v, val] : eq_target) {
                val = val / lead;
            }

            new_eqs.push_back(eq_target);
        }
        else {
            // Should not normally happen, but keep safe fallback
            Equation trivial_eq;
            trivial_eq[var] = Coeff(1);
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

// ------- Helper converters -------

// Convert Python (map<int,int>) to (map<int,Coeff>)
Equation py_to_eq(const std::map<int, int>& py_eq) {
    Equation eq;
    for (const auto& [k, v] : py_eq) {
        eq[k] = Coeff(v);
    }
    return eq;
}

// Convert C++ (map<int,Coeff>) to (map<int,int>)
std::map<int, int> eq_to_py(const Equation& eq) {
    std::map<int, int> py_eq;
    for (const auto& [k, v] : eq) {
        py_eq[k] = v.v;
    }
    return py_eq;
}

// Vector conversions
std::vector<Equation> py_to_eqs(const std::vector<std::map<int,int>>& py_eqs) {
    std::vector<Equation> eqs;
    eqs.reserve(py_eqs.size());
    for (const auto& e : py_eqs) {
        eqs.push_back(py_to_eq(e));
    }
    return eqs;
}

std::vector<std::map<int,int>> eqs_to_py(const std::vector<Equation>& eqs) {
    std::vector<std::map<int,int>> out;
    out.reserve(eqs.size());
    for (const auto& e : eqs) {
        out.push_back(eq_to_py(e));
    }
    return out;
}

// For pair<int, Equation>
std::vector<std::pair<int, std::map<int,int>>> pairs_to_py(
    const std::vector<std::pair<int, Equation>>& vec) {

    std::vector<std::pair<int, std::map<int,int>>> out;
    out.reserve(vec.size());

    for (const auto& [k, eq] : vec) {
        out.emplace_back(k, eq_to_py(eq));
    }

    return out;
}

// ------------ Module -------------

PYBIND11_MODULE(ibp_solver_num, m) {
    m.doc() = "IBP solver using finite fields";

    // --- Wrapped functions ---

    m.def("ibps_to_blocks",
        [](const std::vector<std::map<int,int>>& py_ibps) {

            auto eqs = py_to_eqs(py_ibps);
            auto blocks = ibps_to_blocks(eqs);

            // Convert blocks to Python
            std::vector<std::pair<int, std::vector<std::map<int,int>>>> out;

            for (const auto& [k, vec_eq] : blocks) {
                out.emplace_back(k, eqs_to_py(vec_eq));
            }

            return out;
        },
        "Groups IBP equations"
    );

    m.def("replace_sint",
        [](int var, const std::vector<std::map<int,int>>& py_eqs) {

            auto eqs = py_to_eqs(py_eqs);
            auto result = replace_sint(var, eqs);

            return eqs_to_py(result);
        },
        "Replaces variables in the IBP system"
    );

    m.def("gauss_fwd_elim",
        [](const std::vector<std::pair<int, std::vector<std::map<int,int>>>>& py_blocks, bool verbose) {

            std::vector<Block> blocks;

            for (const auto& [k, vec_eq] : py_blocks) {
                blocks.emplace_back(k, py_to_eqs(vec_eq));
            }

            auto result = gauss_fwd_elim(blocks, verbose);

            return pairs_to_py(result);
        },
        "Gauss forward elimination"
    );

    m.def("gauss_bwd_elim",
        [](const std::vector<std::pair<int, std::map<int,int>>>& py_ibps, bool verbose) {

            std::vector<std::pair<int, Equation>> ibps;

            for (const auto& [k, eq] : py_ibps) {
                ibps.emplace_back(k, py_to_eq(eq));
            }

            auto result = gauss_bwd_elim(ibps, verbose);

            return pairs_to_py(result);
        },
        "Gauss backward elimination"
    );

    m.def("solve_ibps",
        [](const std::vector<std::map<int,int>>& py_ibps, bool verbose) {

            auto eqs = py_to_eqs(py_ibps);
            auto result = solve_ibps(eqs, verbose);

            return pairs_to_py(result);
        },
        "Solves IBP system"
    );
}