#!/usr/bin/env python3
"""
Deinterlaced 24-Class Quantum Clocks — Vibrational D(t) Maps + Hole-Annihilator Field
Each residue class k is its own independent clock (hands k + 90n only).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ====================== USER PATH (adjust if needed) ======================
# Point this to the folder containing April1Sieve2.py on your machine
SCRIPT_DIR = r'C:\Users\jwhel\Downloads'
if os.path.isdir(SCRIPT_DIR):
    sys.path.append(SCRIPT_DIR)
import April1Sieve2 as elder
# ========================================================================

MAX_N = 80000
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

def get_class_holes(k, n_primes):
    holes = []
    amp = np.array(elder.generate_amplitude_map(k, MAX_N))
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5 and is_prime(p):
                holes.append(p)
                if len(holes) >= n_primes:
                    break
    return np.array(holes)

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
t_frames = np.arange(0.1, T_MAX, DT)
D_data = {}

print("Computing vibrational D(t) for each of the 24 class clocks...")
for k in classes:
    holes = get_class_holes(k, PRIMES_PER_CLASS)
    if len(holes) == 0:
        continue
    freqs = np.log(holes)
    base = 1.0 / np.sqrt(holes)
    phases = np.outer(t_frames, freqs) % (2 * np.pi)
    neg_mask = (np.pi/2 < phases) & (phases < 3*np.pi/2)
    R_neg = np.sum(base[None, :] * np.exp(1j * phases) * neg_mask, axis=1)
    R_pos = np.sum(base[None, :] * np.exp(1j * phases) * ~neg_mask, axis=1)
    D = np.abs(R_neg) - np.abs(R_pos)
    peaks, _ = find_peaks(D, prominence=0.08, distance=30)
    D_data[k] = {'D': D, 'bloom_ts': t_frames[peaks], 'bloom_D': D[peaks]}
    print(f"  Class {k:2d} — {len(holes)} primes, {len(peaks)} bloom modes")

# ====================== 6×4 GRID OF INDIVIDUAL CLASS CLOCKS ======================
fig, axes = plt.subplots(6, 4, figsize=(22, 14), dpi=180)
axes = axes.flatten()
for i, k in enumerate(classes):
    if k not in D_data:
        continue
    data = D_data[k]
    ax = axes[i]
    ax.plot(t_frames, data['D'], 'b-', lw=0.9, alpha=0.85)
    ax.plot(data['bloom_ts'], data['bloom_D'], 'r^', markersize=5, alpha=0.9)
    ax.set_title(f'Class {k}  —  D_k(t)', fontsize=10)
    ax.set_xlabel('t', fontsize=8)
    ax.set_ylabel('D(t)', fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)

plt.suptitle('Deinterlaced Vibrational D(t) Maps — 24 Independent Class Clocks\n(each clock uses only hands ≡ k (mod 90))', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('24_class_vibrational_Dt_maps.png', dpi=220)
print("✅ Saved 6×4 grid: 24_class_vibrational_Dt_maps.png")

# ====================== SUMMED VIBRATIONAL FIELD (HOLE ANNIHILATOR) ======================
all_D = np.array([D_data[k]['D'] for k in sorted(D_data.keys())])
sum_D = np.sum(all_D, axis=0)

fig2, ax2 = plt.subplots(figsize=(14, 5), dpi=180)
ax2.plot(t_frames, sum_D, color='#6B2D5C', lw=1.3, label=r'$\sum_k D_k(t)$  (global vibrational sum)')
ax2.axhline(0, color='k', linestyle='--', alpha=0.6, lw=1)
low_mask = sum_D < np.percentile(sum_D, 8)
ax2.fill_between(t_frames, sum_D, where=low_mask, alpha=0.35, color='crimson',
                 label='Positive-momentum suppressed (hole-annihilator zones)')
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\sum_k D_k(t)$')
ax2.set_title('Sum of 24 Vibrational D(t) Maps — Regions Where Positive Momentum Is Annihilated\n(these phase loci force unmarked indices on the number line)')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sum_vibrational_Dt_hole_annihilator.png', dpi=220)
print("✅ Saved summed annihilator field: sum_vibrational_Dt_hole_annihilator.png")

plt.show()