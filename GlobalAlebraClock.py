#!/usr/bin/env python3
"""
Mirror Bloom Symmetry — Algebraic Ideal Holes + Stricter Einselection
Direct D(t) measurement from ground-state-derived operators.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_N = 9000000
N_HANDS = 4500
X_MAX = 350
T_MAX = 760.0
DT = 0.012
# ===================================================

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

# Correct deterministic holes from the algebraic ideal
def get_holes(n_operators, MAX_N=9000000):
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

holes = get_holes(N_HANDS)
print("Algebraic ideal deterministic primes >5 coprime to 90:", holes[:30], "...")

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

# Stricter einselection for negative-plane blooms
peaks_neg, _ = find_peaks(D_neg, prominence=0.25, distance=40)
negative_zeros = t_frames[peaks_neg]

# Positive-plane mirror blooms (unchanged)
peaks_pos, _ = find_peaks(D_pos, prominence=0.08, distance=30)
positive_zeros = t_frames[peaks_pos]

print(f"Negative-plane zeros (stricter): {len(negative_zeros)}")
print(f"Positive-plane mirror zeros: {len(positive_zeros)}")

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
line_neg, = ax1.plot(x, pnt_neg, 'b-', lw=1.8, label='Negative (stricter)')
line_pos, = ax1.plot(x, pnt_pos, 'r-', lw=1.8, label='Positive mirror')
ax1.set_xscale('log')
ax1.set_title('Prime Counting Function Corrections')
ax1.legend()

ax2.plot(x, np.abs(actual_pi - pnt_neg), 'b-', lw=1.8, label='Error (negative stricter)')
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

ani.save('pnt_correction_comparison_negative_stricter_vs_positive_ideal.mp4', writer='ffmpeg', fps=25, dpi=180)
print("✅ Video saved as 'pnt_correction_comparison_negative_stricter_vs_positive_ideal.mp4'")

plt.tight_layout()
plt.show()

print("\nError comparison (final x = {:.0f})".format(x[-1]))
print("Negative zeros (stricter) error: {:.2f}".format(np.abs(actual_pi[-1] - pnt_neg[-1])))
print("Positive mirror zeros error: {:.2f}".format(np.abs(actual_pi[-1] - pnt_pos[-1])))