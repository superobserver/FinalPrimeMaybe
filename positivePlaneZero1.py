#!/usr/bin/env python3
"""
Mirror Bloom Symmetry — Global True Primes >5 coprime to 90
Compares negative-plane vs positive-plane mirror zero corrections to π(x).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

# ====================== CONFIG ======================
MAX_N = 9000000                  # sieve limit
N_HANDS = 4500                   # number of primes for clock hands
X_MAX = 350                     # PNT comparison limit
T_MAX = 760.0                   # adjust this to get more zeroes
DT = 0.012
# ===================================================

def sieve_primes_coprime_90(limit):
    """Standard sieve for all primes >5 coprime to 90."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    # Filter >5 and coprime to 90 (not divisible by 2,3,5)
    primes = primes[primes > 5]
    primes = primes[(primes % 2 != 0) & (primes % 3 != 0) & (primes % 5 != 0)]
    return primes[:N_HANDS]

holes = sieve_primes_coprime_90(MAX_N)
print("Correct global primes >5 coprime to 90:", holes[:30], "...")

# ====================== RESONANCE COMPUTATION ======================
t_frames = np.arange(0.1, T_MAX, DT)
freqs = np.log(holes)
base_lengths = 1.0 / np.sqrt(holes)
phases = np.outer(t_frames, freqs) % (2 * np.pi)

neg_mask = (np.pi/2 < phases) & (phases < 3*np.pi/2)
R_neg = np.sum(base_lengths[None, :] * np.exp(1j * phases) * neg_mask, axis=1)
R_pos = np.sum(base_lengths[None, :] * np.exp(1j * phases) * ~neg_mask, axis=1)
D_neg = np.abs(R_neg) - np.abs(R_pos)
D_pos = np.abs(R_pos) - np.abs(R_neg)

# Negative-plane blooms
peaks_neg, _ = find_peaks(D_neg, prominence=0.08, distance=30)
negative_zeros = t_frames[peaks_neg]

# Positive-plane mirror blooms
peaks_pos, _ = find_peaks(D_pos, prominence=0.08, distance=30)
positive_zeros = t_frames[peaks_pos]

print(f"Negative-plane zeros: {len(negative_zeros)}")
print(f"Positive-plane mirror zeros: {len(positive_zeros)}")
print(f"Negative-plane zeros: {negative_zeros[:100]}")
print(f"Positive-plane mirror zeros: {positive_zeros[:100]}")


# ====================== PNT CORRECTION ======================
x = np.logspace(2, np.log10(X_MAX), 800)
actual_pi = np.array([np.sum(holes < xi) for xi in x])

def explicit_correction(x, zeros):
    corr = np.zeros_like(x, dtype=float)
    for t in zeros:
        if t < 1: continue
        amp = np.sqrt(x) / t
        corr += -amp * np.cos(t * np.log(x))
    return corr

corr_neg = explicit_correction(x, negative_zeros)
corr_pos = explicit_correction(x, positive_zeros)

li_approx = x / np.log(x) + x / (np.log(x)**2)
pnt_neg = li_approx + corr_neg
pnt_pos = li_approx + corr_pos

# ====================== ANIMATION ======================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=200)

ax1.plot(x, actual_pi, 'k-', lw=2, label='Actual π(x)')
line_neg, = ax1.plot(x, pnt_neg, 'b-', lw=1.8, label='Negative-zero correction')
line_pos, = ax1.plot(x, pnt_pos, 'r-', lw=1.8, label='Positive-mirror correction')
ax1.set_xscale('log')
ax1.set_title('Prime Counting Function Corrections')
ax1.legend()

ax2.plot(x, np.abs(actual_pi - pnt_neg), 'b-', lw=1.8, label='Error (negative)')
ax2.plot(x, np.abs(actual_pi - pnt_pos), 'r-', lw=1.8, label='Error (positive mirror)')
ax2.set_xscale('log')
ax2.set_title('Absolute Error Comparison')
ax2.legend()

def animate(frame):
    n = min(frame + 1, len(negative_zeros))
    corr_neg_frame = explicit_correction(x, negative_zeros[:n])
    corr_pos_frame = explicit_correction(x, positive_zeros[:n])
    line_neg.set_ydata(li_approx + corr_neg_frame)
    line_pos.set_ydata(li_approx + corr_pos_frame)
    return line_neg, line_pos

ani = FuncAnimation(fig, animate, frames=max(len(negative_zeros), len(positive_zeros)), interval=40, blit=False, repeat=True)

ani.save('pnt_correction_comparison_negative_vs_positive_global.mp4', writer='ffmpeg', fps=25, dpi=180)
print("✅ Video saved as 'pnt_correction_comparison_negative_vs_positive_global.mp4'")

plt.tight_layout()
plt.show()

print("\nError comparison (final x = {:.0f})".format(x[-1]))
print("Negative zeros error: {:.2f}".format(np.abs(actual_pi[-1] - pnt_neg[-1])))
print("Positive mirror zeros error: {:.2f}".format(np.abs(actual_pi[-1] - pnt_pos[-1])))