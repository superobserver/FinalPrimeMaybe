#!/usr/bin/env python3
"""
Bounded D(t) Envelope Computation — Min/Max D(t) for Zeta Zeros in Finite Pool
Direct computation of the quantized bloom spectrum from any finite operator pool.
"""

import sys
import numpy as np
from scipy.signal import find_peaks

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def get_holes(n_operators, MAX_N=80000):
    holes = []
    classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
    for k in classes:
        amp = elder.generate_amplitude_map(k, MAX_N)
        for m in range(len(amp)):
            if amp[m] == 0:
                p = 90 * m + k
                if p > 5 and is_prime(p):
                    holes.append(p)
                    if len(holes) >= n_operators:
                        return np.array(sorted(set(holes)))
    return np.array(sorted(set(holes)))[:n_operators]

# Compute envelope for a given pool size
def compute_d_envelope(n_operators, T_MAX=120.0, DT=0.008):
    holes = get_holes(n_operators)
    t_frames = np.arange(0.1, T_MAX, DT)
    freqs = np.log(holes)
    base_lengths = 1.0 / np.sqrt(holes)
    phases = np.outer(t_frames, freqs) % (2 * np.pi)
    
    neg_mask = (np.pi/2 < phases) & (phases < 3*np.pi/2)
    R_neg = np.sum(base_lengths[None, :] * np.exp(1j * phases) * neg_mask, axis=1)
    R_pos = np.sum(base_lengths[None, :] * np.exp(1j * phases) * ~neg_mask, axis=1)
    D = np.abs(R_neg) - np.abs(R_pos)
    
    peaks, _ = find_peaks(D, prominence=0.08, distance=30)
    bloom_D_values = D[peaks]
    
    return {
        'pool_size': n_operators,
        'min_D': bloom_D_values.min() if len(bloom_D_values) > 0 else None,
        'max_D': bloom_D_values.max() if len(bloom_D_values) > 0 else None,
        'num_modes': len(bloom_D_values),
        'bloom_D_values': bloom_D_values
    }

# Example computation
print("Computing bounded D(t) envelope for finite pools...\n")
for n in [8, 16, 24, 35, 50]:
    result = compute_d_envelope(n)
    print(f"Pool size {result['pool_size']:2d} → {result['num_modes']} einselected modes")
    print(f"   D_min = {result['min_D']:.4f}   D_max = {result['max_D']:.4f}")
    print(f"   Range width = {result['max_D'] - result['min_D']:.4f}\n")