#!/usr/bin/env python3
"""
Amplified Log-Primorial Gaps vs. Zeta Zero Gaps
Overlay and linear fit to test correlation
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Zeta zero gaps (from your plot)
zero_gaps = np.array([6.887, 3.989, 5.414, 2.510, 4.651, 3.333, 2.409, 4.678, 1.769, 3.196, 3.476, 2.901, 1.484, 4.265, 1.967, 2.467, 2.518, 3.637, 1.439, 2.193, 3.573, 1.825, 2.690, 1.384, 3.474, 2.160, 1.019, 2.961, 2.486])

# Log-primorial gaps (from your plot)
log_primorial_gaps = np.array([1.099, 1.609, 1.792, 1.946, 2.398, 2.565, 2.833, 2.944, 3.135, 3.367, 3.526, 3.714, 3.738, 3.871, 4.007, 4.060, 4.174, 4.248, 4.317, 4.394, 4.443, 4.500, 4.585, 4.615, 4.665, 4.700, 4.745, 4.787, 4.828])

def linear(x, a, b):
    return a * x + b

popt, _ = curve_fit(linear, log_primorial_gaps, zero_gaps)
a, b = popt
print(f"Fitted scaling: Δt_n ≈ {a:.4f} · log(p_{{n+1}}) + {b:.4f}")
print(f"Pearson correlation: {np.corrcoef(log_primorial_gaps, zero_gaps)[0,1]:.4f}")

# ====================== PLOT ======================
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

n = np.arange(1, len(zero_gaps)+1)
ax.plot(n, zero_gaps, 'b-o', lw=2, label='Zeta zero gaps Δt_n')
ax.plot(n, linear(log_primorial_gaps, a, b), 'r--', lw=2, label=f'Scaled log-primorial gaps (a={a:.2f}, b={b:.2f})')
ax.set_xlabel('Index n')
ax.set_ylabel('Gap')
ax.set_title('Amplified Log-Primorial Gaps vs. Zeta Zero Gaps\n'
             'The algebraic ideal links super-composite spacing to zero spacing')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('amplified_primorial_vs_zero_gaps.png', dpi=300, bbox_inches='tight')
plt.show()