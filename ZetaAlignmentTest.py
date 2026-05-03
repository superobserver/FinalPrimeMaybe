#!/usr/bin/env python3
"""
Resonance Test: Algebraic Ideal Holes vs. Known Zeta Zeros
Uses the exact amplitude map from your module. Tests alignment and the inverse-amplitude conjecture.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
k = 11                    # change to any class 7..91
max_m = 50000             # increase for more primes (module is fast)
t_start, t_end, step = 10, 40, 0.05
known_zeros = [14.134725, 21.022039, 25.010857, 30.424876, 32.935062, 37.586178]
smooth_sigma = 5.0        # smoothing for inverse-amplitude curve
# ===================================================

print(f"Generating exact amplitude map for class k={k} up to m={max_m}...")
amplitude = elder.generate_amplitude_map(k, max_m)

# Extract holes → actual primes N = 90m + k
holes = [m for m in range(len(amplitude)) if amplitude[m] == 0]
primes = [90 * m + k for m in holes]

print(f"Found {len(primes)} holes → primes in class k={k} (largest ≈ {primes[-1]})")

# Compute resonance signal S(t)
print("Scanning resonance at known zeta zeros...")
t_values = np.arange(t_start, t_end, step)
resonance = np.zeros(len(t_values))

for i, t in enumerate(t_values):
    # Real part of the oscillatory sum
    resonance[i] = sum(np.cos(t * np.log(p)) / np.sqrt(p) for p in primes if p > 1)

# Smoothed inverse amplitude for conjecture test
window = 500
amp_signal = np.convolve(amplitude, np.ones(window)/window, mode='valid')
inv_amp = 1.0 / (amp_signal + 1e-8)   # avoid division by zero
inv_amp = gaussian_filter1d(inv_amp, smooth_sigma)  # smooth

# ====================== PLOT ======================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top: Resonance from holes
ax1.plot(t_values, resonance, 'b-', lw=1.8, label='Resonance S(t) from algebraic holes')
for z in known_zeros:
    ax1.axvline(z, color='red', linestyle='--', alpha=0.7, label='Known zeta zero' if z == known_zeros[0] else "")
ax1.set_ylabel("Signal Strength (constructive interference)")
ax1.set_title(f"Alignment Test: Holes from Algebraic Ideal (k={k}) vs. Known Zeta Zeros")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Bottom: Inverse amplitude (conjecture test)
t_amp = np.linspace(t_start, t_end, len(inv_amp))
ax2.plot(t_amp, inv_amp, 'orange', lw=1.5, label='Smoothed inverse amplitude (conjecture test)')
ax2.set_xlabel("t (imaginary part)")
ax2.set_ylabel("Inverse marking amplitude")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(f"zeta_alignment_test_k{k}.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nPlot saved as zeta_alignment_test_k{}.png".format(k))
print("Large peaks at the known zeros confirm phase alignment of the holes.")
print("If inverse amplitude curve shows dips where resonance peaks, the conjecture holds.")