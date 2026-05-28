#!/usr/bin/env python3
"""
Comparison of Primorial Gaps vs. Zeta Zero Gaps
Curve fit and correlation between log-primorial gaps and Δt_n
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Known first 30 imaginary parts of non-trivial zeta zeros
known_zeros = np.array([
    14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
    37.5861781588, 40.9187190121, 43.3270732809, 48.0051508812, 49.7738324777,
    52.9703214777, 56.4462476971, 59.3470440027, 60.8317785246, 65.1125440481,
    67.0798105295, 69.5464015971, 72.0671576745, 75.7046906991, 77.1448400689,
    79.3373750202, 82.9103808544, 84.7354929805, 87.4252746131, 88.8096600006,
    92.4918992706, 94.6513440410, 95.8706342282, 98.8311942188, 101.3178510057
])

# Zeta zero gaps Δt_n
zero_gaps = np.diff(known_zeros)

# First 30 primes
primes = np.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,
                   73,79,83,89,97,101,103,107,109,113])

# Log-primorial gaps = diff(log(primorials)) = log(p_{n+1})
log_primorial_gaps = np.diff(np.cumsum(np.log(primes)))

# Linear fit: zero_gap ≈ a * log_primorial_gap + b
def linear(x, a, b):
    return a * x + b

popt, pcov = curve_fit(linear, log_primorial_gaps, zero_gaps)
a, b = popt

print(f"Fitted model: Δt_n ≈ {a:.4f} · log(p_{{n+1}}) + {b:.4f}")
print(f"Correlation coefficient: {np.corrcoef(log_primorial_gaps, zero_gaps)[0,1]:.4f}")

# ====================== PLOT ======================
fig, axs = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=False)

axs[0].plot(np.arange(1, len(zero_gaps)+1), zero_gaps, 'b-o', label='Zeta zero gaps Δt_n')
axs[0].set_ylabel('Gap Δt_n')
axs[0].set_title('Zeta Zero Gaps')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

axs[1].plot(np.arange(1, len(log_primorial_gaps)+1), log_primorial_gaps, 'r-o', label='Log-primorial gaps log(p_{n+1})')
axs[1].set_xlabel('Index n')
axs[1].set_ylabel('Log gap')
axs[1].set_title('Log-Primorial Gaps')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

plt.suptitle('Primorial Gaps vs. Zeta Zero Gaps\n'
             'The algebraic ideal links aperiodic hole distribution to quasi-stochastic zero spacing')
plt.tight_layout()
plt.savefig('primorial_vs_zero_gaps.png', dpi=300, bbox_inches='tight')
plt.show()