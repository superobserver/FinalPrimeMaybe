#!/usr/bin/env python3
"""
Quasi-LCM Harmonic Convergence Search for Zeta Zeros
Uses the algebraic ideal's deterministic holes.
Deinterlaces into 24 class channels and finds the MINIMUM number of classes
required to achieve "sufficient confidence" in each zero.
"""

import sys
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_M = 50000                  # increase for higher zeros
N_CANDIDATES = 8               # how many zeros to discover
CONFIDENCE_THRESHOLD = 0.90    # fraction of full-stack |S(t)| needed
T_START = 10.0
T_STEP = 0.001                 # coarse grid resolution
# ===================================================

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

# Precompute primes per class (deterministic holes)
print("Generating deterministic holes from algebraic ideal...")
primes_per_class = {}
all_primes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_M)
    holes_m = [m for m in range(len(amp)) if amp[m] == 0]
    primes = sorted([90 * m + k for m in holes_m if 90 * m + k > 5])
    primes_per_class[k] = np.array(primes, dtype=float)
    all_primes.extend(primes)
all_primes = np.array(sorted(set(all_primes)), dtype=float)
print(f"Full stack: {len(all_primes)} holes (largest p ≈ {all_primes[-1]:,.0f})")

def compute_S(t, prime_list):
    """Fast vectorized S(t) for any list of primes."""
    if len(prime_list) == 0:
        return 0.0
    return np.sum(np.cos(t * np.log(prime_list)) / np.sqrt(prime_list))

def find_quasi_lcm_zero(t_min, t_max, prime_list, tol=1e-10):
    """Grid search + bounded optimization to locate next zero."""
    t_grid = np.arange(t_min, t_max, T_STEP)
    S_grid = np.array([compute_S(t, prime_list) for t in t_grid])
    t0 = t_grid[np.argmin(S_grid)]
    res = minimize_scalar(lambda t: compute_S(t, prime_list),
                          bounds=(t0 - 2, t0 + 2), method='bounded', tol=tol)
    return res.x, res.fun

# ====================== MAIN SEARCH ======================
print(f"\nSearching for first {N_CANDIDATES} quasi-LCM zeros...")
candidates = []
current_t = T_START

for i in range(N_CANDIDATES):
    # 1. Full-stack reference zero
    t_full, S_full = find_quasi_lcm_zero(current_t, current_t + 20, all_primes)
    print(f"\nZero {i+1:2d} | Full stack → t = {t_full:.8f}   S = {S_full:8.4f}")

    # 2. Progressive class addition (ranked by momentum at approx t)
    momentum = []
    for k in classes:
        S_k = compute_S(t_full, primes_per_class[k])
        momentum.append((k, abs(S_k)))
    momentum.sort(key=lambda x: -x[1])  # strongest first

    # Add classes until confidence threshold is reached
    partial_primes = np.array([], dtype=float)
    min_classes = 0
    for rank, (k, _) in enumerate(momentum):
        partial_primes = np.append(partial_primes, primes_per_class[k])
        S_partial = compute_S(t_full, partial_primes)
        min_classes = rank + 1
        confidence = abs(S_partial) / abs(S_full)
        if confidence >= CONFIDENCE_THRESHOLD:
            break

    print(f"   → Minimum classes required: {min_classes} "
          f"(confidence {confidence*100:.1f}%)")
    print(f"   → Leading classes: {[k for k,_ in momentum[:min_classes]]}")

    candidates.append((t_full, S_full, min_classes))
    current_t = t_full + 0.5   # move past current zero

# ====================== SUMMARY TABLE ======================
print("\n=== Quasi-LCM Convergence Summary ===")
for i, (t, S, ncls) in enumerate(candidates):
    print(f"Zero {i+1:2d} | t = {t:10.6f} | S = {S:8.4f} | min classes = {ncls}")

# ====================== OPTIONAL PLOT ======================
# Plot full vs minimal-class signal for the first zero
t_plot = np.arange(candidates[0][0]-2, candidates[0][0]+2, 0.005)
S_full_plot = np.array([compute_S(t, all_primes) for t in t_plot])

partial_primes_min = np.array([], dtype=float)
for k in [k for k,_ in momentum[:candidates[0][2]]]:
    partial_primes_min = np.append(partial_primes_min, primes_per_class[k])

S_min_plot = np.array([compute_S(t, partial_primes_min) for t in t_plot])

plt.figure(figsize=(12, 6), dpi=300)
plt.plot(t_plot, S_full_plot, 'k-', lw=2, label='Full 24-class stack')
plt.plot(t_plot, S_min_plot, 'r--', lw=2, label=f'Min {candidates[0][2]} classes')
plt.axvline(candidates[0][0], color='blue', linestyle='--', lw=1.5, label='Quasi-LCM zero')
plt.title('LCM-like Convergence: Full Stack vs Minimal Classes (First Zero)')
plt.xlabel('t')
plt.ylabel('S(t)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('quasi_lcm_min_classes_first_zero.png', dpi=300, bbox_inches='tight')
plt.show()