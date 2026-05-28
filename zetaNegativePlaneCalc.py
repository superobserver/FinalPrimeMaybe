#!/usr/bin/env python3
"""
Clock Model — Negative-Space Overlap & Tightened Zero Windows
Calculates entry/exit times and identifies candidate alignment windows
where many operators cluster in negative space with low phase variance.
"""

import sys
import numpy as np
sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ====================== CONFIG ======================
MAX_M = 30000                  # stack size for first few zeros
N_HANDS = 40                   # operators shown
T_SEARCH = 100.0               # search up to this t
NEG_THRESHOLD = 0.65 * N_HANDS # fraction of operators in negative space
# ===================================================

classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
holes = []
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_M)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5:
                holes.append(p)
holes = np.array(sorted(set(holes)))[:N_HANDS]
freqs = np.log(holes)
print(f"Using {len(holes)} operators (largest p ≈ {holes[-1]:,.0f})")

def negative_half(t):
    """Return mask of operators currently in negative half-plane."""
    phases = (t * freqs) % (2 * np.pi)
    return (np.pi/2 < phases) & (phases < 3*np.pi/2)

# Scan for candidate windows
t_candidates = np.arange(0.0, T_SEARCH, 0.01)
overlap = np.array([np.sum(negative_half(t)) for t in t_candidates])
variance = []
for t in t_candidates:
    phases = (t * freqs) % (2 * np.pi)
    neg_phases = phases[negative_half(t)]
    if len(neg_phases) > 1:
        variance.append(np.var(neg_phases))
    else:
        variance.append(np.inf)
variance = np.array(variance)

# Candidate windows: high overlap + low variance
high_overlap = overlap > NEG_THRESHOLD
low_var_idx = np.where((high_overlap) & (variance < 0.5))[0]
candidate_windows = []
for idx in low_var_idx:
    t = t_candidates[idx]
    candidate_windows.append((t, overlap[idx], variance[idx]))

print("\nCandidate alignment windows (tightened ranges for zero search):")
for t, ov, var in sorted(candidate_windows)[:12]:
    print(f"  t ≈ {t:8.4f} | negative operators: {ov:3.0f}/{N_HANDS} | phase var: {var:6.3f}")

# The true zeros lie inside these windows; refine with quasi-LCM search inside them