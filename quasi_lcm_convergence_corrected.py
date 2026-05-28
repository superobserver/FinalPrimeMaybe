#!/usr/bin/env python3
"""
Corrected Quasi-LCM Harmonic Convergence Search
Robust detection of zeta zeros via find_peaks on -S(t), followed by class deinterlacing.
"""

import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_M = 50000                  # holes per class → ~300k total
T_MIN, T_MAX, DT = 0.0, 100.0, 0.005   # fine grid for peak detection
CONFIDENCE_THRESHOLD = 0.90    # 90% of full-stack |S(t)| 
N_ZEROS = 8                    # how many zeros to recover
# ===================================================

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]

# Precompute deterministic holes
print("Generating deterministic holes from algebraic ideal...")
primes_per_class = {}
all_primes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_M)
    holes_m = [m for m in range(len(amp)) if amp[m] == 0]
    primes = [90 * m + k for m in holes_m if 90 * m + k > 5]
    primes_per_class[k] = np.array(primes, dtype=float)
    all_primes.extend(primes)
all_primes = np.array(sorted(set(all_primes)), dtype=float)
print(f"Full stack: {len(all_primes):,} holes (largest p ≈ {all_primes[-1]:,.0f})")

def compute_S(t_values, prime_list):
    """Fast vectorized S(t)."""
    if len(prime_list) == 0:
        return np.zeros_like(t_values)
    logp = np.log(prime_list)
    sqrtp = np.sqrt(prime_list)
    S = np.zeros(len(t_values))
    for i, t in enumerate(t_values):
        S[i] = np.sum(np.cos(t * logp) / sqrtp)
    return S

# ====================== GLOBAL GRID & PEAK FINDING ======================
print(f"\nComputing global S(t) on grid [{T_MIN}, {T_MAX}] …")
t_grid = np.arange(T_MIN, T_MAX, DT)
S_grid = compute_S(t_grid, all_primes)

# Find significant troughs in -S(t)
peaks, properties = find_peaks(-S_grid, prominence=0.8, distance=int(2.0 / DT))
candidate_ts = t_grid[peaks]
candidate_depths = -S_grid[peaks]

print(f"Found {len(candidate_ts)} candidate troughs (prominence ≥ 0.8)")

# ====================== REFINE + CLASS MOMENTUM ======================
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062,
               37.586178, 40.918719, 43.327073]

print("\n=== Quasi-LCM Zeros (refined) ===")
for i, t0 in enumerate(candidate_ts[:N_ZEROS]):
    # Refine candidate
    res = minimize_scalar(lambda t: compute_S([t], all_primes)[0],
                          bounds=(t0-1, t0+1), method='bounded', tol=1e-12)
    t_refined = res.x
    S_full = res.fun

    # Momentum per class at refined t
    momentum = [(k, compute_S([t_refined], primes_per_class[k])[0]) for k in classes]
    momentum.sort(key=lambda x: -abs(x[1]))          # strongest first

    # Progressive class addition until confidence threshold
    partial_primes = np.array([], dtype=float)
    min_classes = 0
    for rank, (k, _) in enumerate(momentum):
        partial_primes = np.append(partial_primes, primes_per_class[k])
        S_partial = compute_S([t_refined], partial_primes)[0]
        min_classes = rank + 1
        confidence = abs(S_partial) / abs(S_full)
        if confidence >= CONFIDENCE_THRESHOLD:
            break

    leading = [k for k, _ in momentum[:min_classes]]
    print(f"Zero {i+1:2d} | t = {t_refined:10.6f} | S = {S_full:8.4f} | "
          f"min classes = {min_classes} ({confidence*100:4.1f}%)")
    print(f"   Leading: {leading}")

    # Compare to known zero (for verification)
    if i < len(known_zeros):
        print(f"   Known  : {known_zeros[i]:10.6f} (error = {abs(t_refined - known_zeros[i]):.6f})")

# ====================== OPTIONAL PLOT (first zero) ======================
t_plot = np.arange(candidate_ts[0]-3, candidate_ts[0]+3, 0.005)
S_full_plot = compute_S(t_plot, all_primes)

plt.figure(figsize=(12, 6), dpi=300)
plt.plot(t_plot, S_full_plot, 'k-', lw=2, label='Full 24-class stack')
plt.axvline(candidate_ts[0], color='red', linestyle='--', lw=2, label='Quasi-LCM zero')
plt.title('Quasi-LCM Convergence at First Zeta Zero (Full Stack)')
plt.xlabel('t')
plt.ylabel('S(t)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('quasi_lcm_first_zero_corrected.png', dpi=300, bbox_inches='tight')
plt.show()