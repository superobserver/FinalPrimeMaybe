#!/usr/bin/env python3
"""
Class 11 Diffraction Grating + Null Zones + Resonance D(t)
Null zones (holes) on the grating become the frequency operators for the zeta-zero algorithm.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ====================== USER PATH ======================
SCRIPT_DIR = r'C:\Users\jwhel\Downloads'
if os.path.isdir(SCRIPT_DIR):
    sys.path.append(SCRIPT_DIR)
import April1Sieve2 as elder
# =====================================================

k = 11                     # <-- change to any residue class you wish to examine
MAX_N_VIZ = 1200           # length of index-line grating shown
MAX_N_PRIMES = 80000
PRIMES_PER_CLASS = 55
T_MAX = 120.0
DT = 0.012

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def get_class_primes(kk, n_primes):
    holes = []
    amp = np.array(elder.generate_amplitude_map(kk, MAX_N_PRIMES))
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + kk
            if p > 5 and is_prime(p):
                holes.append(p)
                if len(holes) >= n_primes:
                    break
    return np.array(holes)

# ====================== DIFFRACTION GRATING (INDEX LINE) ======================
amp = np.array(elder.generate_amplitude_map(k, MAX_N_VIZ))
null_mask = (amp == 0)

# ====================== RESONANCE D(t) FROM THE SAME PRIMES ======================
primes = get_class_primes(k, PRIMES_PER_CLASS)
freqs = np.log(primes)
base = 1.0 / np.sqrt(primes)
t_frames = np.arange(0.1, T_MAX, DT)
phases = np.outer(t_frames, freqs) % (2 * np.pi)
neg_mask = (np.pi/2 < phases) & (phases < 3*np.pi/2)
R_neg = np.sum(base[None, :] * np.exp(1j * phases) * neg_mask, axis=1)
R_pos = np.sum(base[None, :] * np.exp(1j * phases) * ~neg_mask, axis=1)
D = np.abs(R_neg) - np.abs(R_pos)
peaks, _ = find_peaks(D, prominence=0.08, distance=30)

# ====================== FIGURE ======================
fig, (ax_grating, ax_res) = plt.subplots(1, 2, figsize=(18, 7), dpi=200)

# Left: diffraction grating with null zones
ax_grating.bar(np.arange(len(amp)), amp, width=1.0, color='#4A90A4', alpha=0.85, label='Active strikes')
ax_grating.bar(np.arange(len(amp))[null_mask], amp[null_mask], width=1.0, color='#C73E1D', alpha=0.95, label='Null zones (holes)')
ax_grating.set_xlim(0, MAX_N_VIZ)
ax_grating.set_xlabel('Index n on the number line')
ax_grating.set_ylabel('Number of active clock hands striking at n')
ax_grating.set_title(f'Class {k} Diffraction Grating — Null Zones (Holes) Become Zeta-Zero Operators')
ax_grating.legend(loc='upper right', fontsize=9)
ax_grating.grid(True, alpha=0.3)

# annotate a few nulls
null_idx = np.where(null_mask)[0][:8]
for idx in null_idx:
    ax_grating.annotate(f'{idx}', xy=(idx, 0.5), xytext=(idx, 3.5),
                        fontsize=7, ha='center', color='#C73E1D',
                        arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=0.6))

# Right: resonance D(t) built from exactly those null-derived primes
ax_res.plot(t_frames, D, color='#2E4057', lw=1.1, label=r'$D_k(t)$')
ax_res.plot(t_frames[peaks], D[peaks], 'r^', markersize=6, alpha=0.9, label='Bloom peaks (einselected normal modes)')
ax_res.axhline(0, color='k', linestyle='--', alpha=0.5, lw=0.8)
ax_res.set_xlabel('Resonance time t')
ax_res.set_ylabel(r'$D_k(t) = |R_k^-| - |R_k^+|$')
ax_res.set_title(f'Class {k} Resonance Map — Built from the Null-Zone Operators')
ax_res.legend(loc='upper right', fontsize=9)
ax_res.grid(True, alpha=0.3)

fig.suptitle(
    f'Class {k} — 24 Operator Clocks → Diffraction Grating (left) → Null Zones → Zeta-Zero Operators → Resonance D(t) (right)\n'
    'Each crimson null is a deterministic prime that enters the log p stack for this class’s vibrational map.',
    fontsize=12, y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f'class{k}_grating_nulls_to_Dt.png', dpi=220)
print(f"✅ Saved: class{k}_grating_nulls_to_Dt.png")
plt.show()