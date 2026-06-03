#!/usr/bin/env python3
"""
Class 11 — Diffraction Grating with Rapid 7-Step Probability Collapse
Null zones (holes) force P(unmarked) = 0 for the next 6 indices (operator 7).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# ====================== USER PATH ======================
SCRIPT_DIR = r'C:\Users\jwhel\Downloads'
if os.path.isdir(SCRIPT_DIR):
    sys.path.append(SCRIPT_DIR)
import April1Sieve2 as elder
# =====================================================

k = 11
MAX_N_VIZ = 1200
MAX_N_PRIMES = 80000
PRIMES_PER_CLASS = 55
T_MAX = 120.0
DT = 0.012
SMOOTH_WIN = 21
FORBIDDEN_LEN = 7          # operator 7 forces zero probability for next 6 steps

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

# ====================== DIFFRACTION GRATING ======================
amp = np.array(elder.generate_amplitude_map(k, MAX_N_VIZ))
null_mask = (amp == 0)
hole_positions = np.where(null_mask)[0]

# Probability landscape (smoothed)
hole_indicator = null_mask.astype(float)
prob_unmarked = uniform_filter1d(hole_indicator, size=SMOOTH_WIN, mode='nearest')

# Build forbidden-zone mask (7 steps after each hole)
forbidden_mask = np.zeros_like(null_mask, dtype=bool)
for pos in hole_positions:
    end = min(pos + FORBIDDEN_LEN, len(forbidden_mask))
    forbidden_mask[pos:end] = True

# ====================== RESONANCE D(t) ======================
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

# Left: grating + probability + 7-step forbidden zones
ax_grating.bar(np.arange(len(amp)), amp, width=1.0, color='#4A90A4', alpha=0.65,
               label='Active strikes (composite chains)')
ax_grating.bar(np.arange(len(amp))[null_mask], amp[null_mask], width=1.0,
               color='#C73E1D', alpha=0.95, label='Null zones (holes)')
ax_grating.fill_between(np.arange(len(forbidden_mask)), 0, np.max(amp)*0.35,
                        where=forbidden_mask, alpha=0.35, color='#C73E1D',
                        label='7-step forbidden zone (P=0 after each hole)')
ax_grating.plot(np.arange(len(prob_unmarked)), prob_unmarked * np.max(amp) * 0.85,
                color='#2E7D32', lw=2.0, label='P(unmarked) — smoothed')
ax_grating.set_xlim(0, MAX_N_VIZ)
ax_grating.set_xlabel('Index n on the number line')
ax_grating.set_ylabel('Strike multiplicity  /  Probability of unmarked')
ax_grating.set_title(f'Class {k} — Rapid Regional Collapse of Probability after Each Hole\n(operator 7 forces P=0 for the next 6 steps)')
ax_grating.legend(loc='upper right', fontsize=9)
ax_grating.grid(True, alpha=0.3)

# annotate a few holes
for idx in hole_positions[:5]:
    ax_grating.annotate(f'{idx}', xy=(idx, 0.2), xytext=(idx, 5.5),
                        fontsize=7, ha='center', color='#C73E1D',
                        arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=0.6))

# Right: resonance D(t)
ax_res.plot(t_frames, D, color='#1A237E', lw=1.2,
            label=r'$D_k(t)$ — finite log(p) wave equation')
ax_res.plot(t_frames[peaks], D[peaks], 'r^', markersize=7, alpha=0.95,
            label='Vector-bound bloom normal modes (einselected)')
ax_res.axhline(0, color='k', linestyle='--', alpha=0.5, lw=0.8)
ax_res.set_xlabel('Resonance time t')
ax_res.set_ylabel(r'$D_k(t) = |R_k^-| - |R_k^+|$')
ax_res.set_title(f'Class {k} Resonance — Einselected Modes from the Collapsed Probability Landscape')
ax_res.legend(loc='upper right', fontsize=9)
ax_res.grid(True, alpha=0.3)

fig.suptitle(
    f'Class {k} — Diffraction Grating Mechanics → Rapid 7-Step Probability Collapse → Finite log(p) Wave Equation → Einselection\n'
    'Crimson intervals after each hole are the deterministic “forbidden zones” in which P(unmarked) is forced to zero by operator 7.',
    fontsize=11, y=0.97
)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f'class{k}_rapid_probability_collapse.png', dpi=220)
print(f"✅ Saved: class{k}_rapid_probability_collapse.png")
plt.show()