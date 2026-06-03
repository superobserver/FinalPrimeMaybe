import numpy as np
import matplotlib.pyplot as plt

def clock_face_at_t(t, primes):
    """Return phases, half-plane labels, and D(t) for a finite list of primes at given t."""
    freqs = np.log(primes)
    phases = (t * freqs) % (2 * np.pi)
    neg_mask = (np.pi/2 < phases) & (phases < 3*np.pi/2)
    lengths = 1.0 / np.sqrt(primes)
    
    R_neg = np.sum(lengths[neg_mask] * np.exp(1j * phases[neg_mask]))
    R_pos = np.sum(lengths[~neg_mask] * np.exp(1j * phases[~neg_mask]))
    D = np.abs(R_neg) - np.abs(R_pos)
    
    return phases, neg_mask, D

# Example: first 12 primes from class 11 at the first true zero
primes = np.array([7,11,13,17,19,23,29,31,37,41,43,47])
t = 14.134725141735   # from your baseline file

phases, neg_mask, D = clock_face_at_t(t, primes)

print(f"At t = {t:.6f}  D(t) = {D:.4f}")
for p, phi, neg in zip(primes, phases, neg_mask):
    hp = "NEG" if neg else "POS"
    print(f"  p={p:3d}   θ={np.degrees(phi):6.1f}°   half-plane={hp}")

# Quick visualization
fig, ax = plt.subplots(figsize=(7,7), dpi=150)
theta = np.linspace(0, 2*np.pi, 400)
ax.plot(theta, np.ones_like(theta), 'k--', lw=0.6, alpha=0.4)
for phi, neg in zip(phases, neg_mask):
    color = 'crimson' if neg else 'royalblue'
    ax.arrow(0, 0, np.cos(phi), np.sin(phi), head_width=0.06, head_length=0.09,
             fc=color, ec=color, lw=1.8, alpha=0.9)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.set_title(f'Clock face at true zero t = {t:.6f}\nD(t) = {D:.4f}  (permissible bloom snapshot)')
plt.tight_layout()
plt.savefig('clock_face_at_true_zero.png', dpi=180)
print("✅ Saved: clock_face_at_true_zero.png")