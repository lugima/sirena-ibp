#pragma once // Evita duplicados en la compilación

#include <flint/fmpz_poly.h>

class FlintPoly {
public:
    fmpz_poly_t poly;

    // 1. Constructor por defecto
    FlintPoly() { 
        fmpz_poly_init(poly); 
    }

    // 2. Destructor
    ~FlintPoly() { 
        fmpz_poly_clear(poly); 
    }

    // 3. Constructor de copia
    FlintPoly(const FlintPoly& other) {
        fmpz_poly_init(poly);
        fmpz_poly_set(poly, other.poly);
    }

    // 4. Constructor desde un número entero (Para poder hacer FlintPoly(1) o FlintPoly(-1))
    FlintPoly(long val) {
        fmpz_poly_init(poly);
        fmpz_poly_set_si(poly, val); // set_si = Set Signed Integer
    }

    // Constructor desde un std::vector (lista de coeficientes)
    // El índice de la lista es el exponente. Ej: {1, 1} -> 1 + x
    FlintPoly(const std::vector<long>& coeffs) {
        fmpz_poly_init(poly);
        for (size_t i = 0; i < coeffs.size(); ++i) {
            fmpz_poly_set_coeff_si(poly, i, coeffs[i]);
        }
    }

    // 5. Asignación (A = B)
    FlintPoly& operator=(const FlintPoly& other) {
        if (this != &other) {
            fmpz_poly_set(poly, other.poly);
        }
        return *this;
    }

    // ---------------- OPERADORES MATEMÁTICOS ----------------

    // Multiplicación (A * B)
    FlintPoly operator*(const FlintPoly& other) const {
        FlintPoly result;
        fmpz_poly_mul(result.poly, poly, other.poly);
        return result;
    }

    // Resta (A - B)
    FlintPoly operator-(const FlintPoly& other) const {
        FlintPoly result;
        fmpz_poly_sub(result.poly, poly, other.poly);
        return result;
    }

    // División exacta (A / B)
    FlintPoly operator/(const FlintPoly& other) const {
        FlintPoly result;
        fmpz_poly_div(result.poly, poly, other.poly); // Usamos división estándar (cociente)
        return result;
    }

    // ---------------- MÉTODOS ÚTILES ----------------

    // Comprobar si es cero
    bool is_zero() const {
        return fmpz_poly_is_zero(poly);
    }

    // Convierte el polinomio a un string legible usando "x" como variable
    std::string to_string() const {
        // FLINT crea un texto en C (char*)
        char* str_c = fmpz_poly_get_str_pretty(poly, "x");
        
        // Lo convertimos al formato de string seguro de C++
        std::string resultado(str_c);
        
        // ¡VITAL! Liberamos la memoria de FLINT para no tener fugas (memory leaks)
        flint_free(str_c); 
        
        return resultado;
    }

    // Extrae los coeficientes y los devuelve como un std::vector (lista)
    // El formato devuelto es [x^0, x^1, x^2, ...]
    std::vector<long> get_coeffs() const {
        // Obtenemos la longitud del polinomio (cuántos coeficientes tiene)
        long len = fmpz_poly_length(poly);
        
        std::vector<long> coeffs(len);
        for (long i = 0; i < len; ++i) {
            // fmpz_poly_get_coeff_si saca el coeficiente como un entero largo (long)
            coeffs[i] = fmpz_poly_get_coeff_si(poly, i);
        }
        
        return coeffs;
    }
};

// Función MCD externa (pero dentro del mismo header)
inline FlintPoly gcd(const FlintPoly& a, const FlintPoly& b) {
    FlintPoly result;
    fmpz_poly_gcd(result.poly, a.poly, b.poly);
    return result;
}