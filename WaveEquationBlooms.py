#!/usr/bin/env python3
"""
Finite-Pool Bloom Spectrum Computation — Wave-Equation Partitioning of t-Span
Direct visualization of the quantized normal modes for any finite operator pool.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

# Compute bloom spectrum for a finite pool
def compute_bloom_spectrum(n_operators, T_MAX=120.0, DT=0.008):
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
    bloom_ts = t_frames[peaks]
    bloom_D = D[peaks]
    
    return {
        'pool_size': n_operators,
        't_span': (t_frames[0], t_frames[-1]),
        'num_modes': len(bloom_ts),
        'bloom_ts': bloom_ts,
        'bloom_D': bloom_D,
        'min_D': bloom_D.min() if len(bloom_D) > 0 else None,
        'max_D': bloom_D.max() if len(bloom_D) > 0 else None
    }

# Example
print("Computing bloom spectrum for finite pools...\n")
for n in [8, 16, 24, 35]:
    result = compute_bloom_spectrum(n)
    print(f"Pool size {result['pool_size']:2d} — t-span [{result['t_span'][0]:.2f}, {result['t_span'][1]:.2f}]")
    print(f"   {result['num_modes']} einselected bloom modes")
    print(f"   D range: [{result['min_D']:.4f}, {result['max_D']:.4f}]")
    print(f"   First few bloom t-values: {result['bloom_ts'][:5]}\n")

# Visualization script (optional — run separately if desired)
# It produces 'finite_pool_bloom_partitioning.mp4' showing the t-span partitioning.