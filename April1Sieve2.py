"""
Module: elder_sieve.py
Corrected computational realization of the 24-class system of ideals I_k.

Generates:
- Amplitude map: multiplicity of markings per index (how many chains hit n)
- Raw index table: sorted list of all marked composites (holes = missing entries)
- Grating data for 2D visualization in the (x, n)-plane

Author: J. W. Helkenberg & Grok (xAI)
Date: March 28, 2026
"""

from collections import defaultdict
import numpy as np

COPRIME_RESIDUES = [1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59, 61, 67, 71, 73, 77, 79, 83, 89]

def get_pairs_for_class(k, modulus=90, use_91=True):
    """Return the 24 (z_eff, o_eff) pairs for class k. Maps 1 → 91."""
    pairs = []
    seen = set()
    for z in COPRIME_RESIDUES:
        try:
            o = (k * pow(z, -1, modulus)) % modulus
            if o not in COPRIME_RESIDUES:
                continue
            pair = tuple(sorted([z, o]))
            if pair in seen:
                continue
            seen.add(pair)
            z_eff = 91 if z == 1 and use_91 else z
            o_eff = 91 if o == 1 and use_91 else o
            pairs.append((z_eff, o_eff))
        except ValueError:
            continue
    return pairs

def generate_amplitude_map(k, max_n):
    """
    Returns amplitude array of length max_n+1.
    amplitude[n] = number of different chains (operators + ribbons) that mark index n.
    """
    pairs = get_pairs_for_class(k)
    amplitude = [0] * (max_n + 1)
    for z, o in pairs:
        l = 180 - (z + o)
        m = 90 - (z + o) + (z * o // 90)
        max_ribbon = int(max_n**0.5) + 10
        for x in range(1, max_ribbon + 1):
            y = 90 * x * x - l * x + m
            if 0 < y <= max_n:
                amplitude[y] += 1
            p = z + 90 * (x - 1)
            q = o + 90 * (x - 1)
            m_step = 1
            while True:
                n_p = y + m_step * p
                if n_p > max_n:
                    break
                amplitude[n_p] += 1
                m_step += 1
            m_step = 1
            while True:
                n_q = y + m_step * q
                if n_q > max_n:
                    break
                amplitude[n_q] += 1
                m_step += 1
    return amplitude

def generate_raw_index_table(k, max_x):
    """Returns sorted list of all unique marked composites up to the epoch."""
    pairs = get_pairs_for_class(k)
    composites = set()
    max_n = int(180 * max_x)
    for z, o in pairs:
        l = 180 - (z + o)
        m = 90 - (z + o) + (z * o // 90)
        for x in range(1, max_x + 1):
            y = 90 * x * x - l * x + m
            if y > 0:
                composites.add(y)
            p = z + 90 * (x - 1)
            q = o + 90 * (x - 1)
            m_step = 1
            while True:
                n_p = y + m_step * p
                if n_p > max_n:
                    break
                composites.add(n_p)
                m_step += 1
            m_step = 1
            while True:
                n_q = y + m_step * q
                if n_q > max_n:
                    break
                composites.add(n_q)
                m_step += 1
    return sorted(composites)

def get_grating_data(k, max_x=20):
    """Returns list of (x, y0, p, q) for plotting the 2D grating."""
    pairs = get_pairs_for_class(k)
    data = []
    for z, o in pairs:
        l = 180 - (z + o)
        m = 90 - (z + o) + (z * o // 90)
        for x in range(1, max_x + 1):
            y0 = 90 * x * x - l * x + m
            p = z + 90 * (x - 1)
            q = o + 90 * (x - 1)
            data.append((x, y0, p, q))
    return data

# Example usage
if __name__ == "__main__":
    k = 11
    max_x = 10
    max_n = 2000

    amp = generate_amplitude_map(k, max_n)
    raw_table = generate_raw_index_table(k, max_x)

    print(f"Amplitude map for class {k} (first 30 indices): {amp[1:31]}")
    print(f"Raw composites (first 20): {raw_table[:20]}")
    holes_first_100 = [i for i in range(1, 101) if i not in raw_table]
    print(f"Holes in first 100 indices: {holes_first_100}")