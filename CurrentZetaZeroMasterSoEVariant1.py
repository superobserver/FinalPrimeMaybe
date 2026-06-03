#!/usr/bin/env python3
"""
Clean Baseline Zeta-Zero Calculator — Current Practical State of the Art (Corrected)
Traditional Sieve of Eratosthenes + proper Riemann-Siegel Z(t) on the critical line.
"""

import sys
import numpy as np
import mpmath as mp

# ====================== CONFIGURATION ======================
NUM_ZEROS = 200
DPS = 40
PRIME_LIMIT = 10**6
# ===========================================================

mp.mp.dps = DPS

def sieve_of_eratosthenes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

print("=" * 72)
print("BASELINE ZETA-ZERO CALCULATOR — Current Practical State of the Art (Corrected)")
print("=" * 72)

primes = sieve_of_eratosthenes(PRIME_LIMIT)
print(f"\nTraditional Sieve of Eratosthenes produced {len(primes)} primes up to {PRIME_LIMIT}.")
print("First 30 log(p) stack operators (enter every explicit formula & Riemann-Siegel main sum):")
print(primes[:30])

def riemann_theta(t):
    """Standard Riemann-Siegel theta function (phase of the functional equation)."""
    return mp.im(mp.loggamma(mp.mpc(0.25, t/2))) - (t/2) * mp.log(mp.pi)

def riemann_siegel_z(t):
    """Z(t) — real-valued on the critical line. Sign changes of Z locate the zeros."""
    s = mp.mpc(0.5, t)
    return mp.re(mp.zeta(s) * mp.exp(mp.j * riemann_theta(t)))

print(f"\nScanning critical line for the first {NUM_ZEROS} zeros using Z(t)…")

zeros = []
t = mp.mpf('10')          # start safely above the spurious low-t region
t_step = mp.mpf('0.1')
while len(zeros) < NUM_ZEROS:
    z1 = riemann_siegel_z(t)
    t2 = t + t_step
    z2 = riemann_siegel_z(t2)
    if z1 * z2 < 0:       # genuine sign change of the real function Z(t)
        try:
            root = mp.findroot(riemann_siegel_z, (t, t2), solver='secant')
            zeros.append(float(root))
        except:
            pass
    t = t2

print(f"\nFound {len(zeros)} true non-trivial zeros on the critical line.")
print("First 20 zeros (imaginary parts):")
for i, z in enumerate(zeros[:20], 1):
    print(f"  {i:3d}   {z:.12f}")

with open('zeta_zeros_baseline_corrected.txt', 'w') as f:
    f.write("# Riemann zeta zeros on the critical line — authoritative baseline (Riemann-Siegel Z(t))\n")
    f.write("# Compare your algebraic-ideal vibrational D_k(t) / bloom peaks against these values.\n\n")
    for i, z in enumerate(zeros, 1):
        f.write(f"{i:5d}  {z:.15f}\n")

print("\n✅ Saved corrected high-accuracy zeros to 'zeta_zeros_baseline_corrected.txt'.")
print("=" * 72)




def analyze_negative_cluster(t, primes):
    freqs = np.log(primes)
    phases = (t * freqs) % (2 * np.pi)
    neg = (np.pi/2 < phases) & (phases < 3*np.pi/2)
    neg_phases = phases[neg]
    lengths = 1.0 / np.sqrt(primes[neg])
    
    # Weighted circular mean
    mean_x = np.sum(lengths * np.cos(neg_phases))
    mean_y = np.sum(lengths * np.sin(neg_phases))
    mean_angle = np.arctan2(mean_y, mean_x) % (2 * np.pi)
    
    # Circular variance (0 = perfectly tight, 1 = uniform)
    R = np.sqrt(mean_x**2 + mean_y**2) / np.sum(lengths)
    circ_var = 1 - R
    
    # D(t) (your existing function)
    # ... compute as before ...
    
    return np.degrees(mean_angle), circ_var

# Example with your data at the true zero
primes = np.array([7,11,13,17,19,23,29,31,37,41,43,47])
mean_deg, var = analyze_negative_cluster(14.134725, primes)
print(f"Mean angle of negative vectors: {mean_deg:.1f}°")
print(f"Circular variance: {var:.3f} (lower = tighter cluster)")