#!/usr/bin/env python3
"""
Alignment Test: Holes from the algebraic ideal vs. known zeta zeros
Reconstructs actual primes N = 90m + k and computes oscillatory contribution.
"""

import sys
import numpy as np
import mpmath

# Load your exact module
sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
k = 11                    # any class 7..91
max_m = 50000             # increase for more primes (module is fast)
# First 6 imaginary parts of known zeta zeros (Im(ρ))
t_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062, 37.586178]
# ===================================================

print(f"Generating amplitude map for class k={k} up to m={max_m}...")
amplitude = elder.generate_amplitude_map(k, max_m)

# Extract holes (unmarked indices = primes in 90m + k)
holes = [m for m in range(len(amplitude)) if amplitude[m] == 0]
primes = [90 * m + k for m in holes]

print(f"Found {len(primes)} holes → primes in class k={k} (largest ≈ {primes[-1]})")

# Compute oscillatory contribution at each known zero
print("\n{:>8} {:>12} {:>12} {:>12}".format("t", "Re(S(t))", "Im(S(t))", "|S(t)|"))
print("-" * 50)

for t in t_zeros:
    S = 0j
    for p in primes:
        if p > 1:
            log_p = np.log(p)
            contrib = (log_p / np.sqrt(p)) * np.exp(-1j * t * log_p)
            S += contrib
    re_S = np.real(S)
    im_S = np.imag(S)
    abs_S = np.abs(S)
    print(f"{t:8.3f} {re_S:12.4f} {im_S:12.4f} {abs_S:12.4f}")

print("\nDone. Large |S(t)| at these t indicates constructive interference from the holes,")
print("exactly as expected if the zeros lie on the critical line.")
 