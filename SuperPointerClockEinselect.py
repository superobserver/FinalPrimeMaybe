import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpmath import mp, log  # high-precision log for phases

# Load your algebraic ideal module
sys.path.append('.')  # adjust if needed
import April1Sieve2 as elder

mp.dps = 50  # high precision

# ===================================================================
# 1. Generate deterministic holes (robust version matching your module)
# ===================================================================
def get_holes(n_operators=10):
    """Collect first n_operators deterministic primes >5 from all 24 classes."""
    holes = []
    max_n = 2000  # adjust if your function uses a different name
    for k in elder.COPRIME_RESIDUES:
        amp = elder.generate_amplitude_map(k, max_n=max_n)  # changed to max_n
        for m in range(len(amp)):
            if amp[m] == 0:
                p = 90 * m + k
                if p > 5:
                    holes.append(p)
                    if len(holes) >= n_operators:
                        return np.array(holes)
    return np.array(holes)

# ===================================================================
# 2. Compute V(t) path for a given pool
# ===================================================================
def compute_v_path(holes, t_max=120.0, dt=0.002):
    t = np.arange(0, t_max + dt, dt)
    phases = np.zeros((len(holes), len(t)), dtype=complex)
    for j, p in enumerate(holes):
        phases[j] = (t * float(mp.log(p))) % (2 * np.pi)
    
    amps = 1.0 / np.sqrt(holes)[:, None]
    neg_mask = (np.pi/2 < phases) & (phases < 3*np.pi/2)
    R_neg = np.sum(amps * np.exp(1j * phases) * neg_mask, axis=0)
    R_pos = np.sum(amps * np.exp(1j * phases) * (~neg_mask), axis=0)
    
    D = np.abs(R_neg) - np.abs(R_pos)
    dV = (R_neg - R_pos) * dt
    V = np.cumsum(dV)
    
    # Super-pointer candidates (bloom modes)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(D, prominence=0.08, distance=30)
    super_pointers = t[peaks]
    
    return t, V, D, super_pointers

# ===================================================================
# 3. Run for small and expanded pools
# ===================================================================
small_pool = get_holes(8)
expanded_pool = get_holes(16)

print("Small pool (8 operators):", small_pool)
print("Expanded pool (16 operators):", expanded_pool)

t_small, V_small, D_small, sp_small = compute_v_path(small_pool)
t_exp, V_exp, D_exp, sp_exp = compute_v_path(expanded_pool)

# ===================================================================
# 4. Visualization: live murmuration + hierarchical V(t) paths
# ===================================================================
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2)

ax_clock = fig.add_subplot(gs[0, 0])   # live clock face (small pool)
ax_v = fig.add_subplot(gs[0, 1])       # V(t) path comparison
ax_d_small = fig.add_subplot(gs[1, 0])
ax_d_exp = fig.add_subplot(gs[1, 1])

# Clock face
theta = np.linspace(0, 2*np.pi, 1000)
ax_clock.plot(np.cos(theta), np.sin(theta), 'k-', lw=1, alpha=0.3)
ax_clock.axvline(0, color='gray', ls='--', alpha=0.5)
ax_clock.axhline(0, color='gray', ls='--', alpha=0.5)
ax_clock.set_xlim(-1.3, 1.3)
ax_clock.set_ylim(-1.3, 1.3)
ax_clock.set_aspect('equal')
ax_clock.set_title('Clock-face murmuration (small pool)')

# V(t) hierarchical comparison
line_small, = ax_v.plot(V_small.real, V_small.imag, 'r-', lw=2, alpha=0.8, label='Small pool (8 ops) — recurrent loop')
line_exp, = ax_v.plot(V_exp.real, V_exp.imag, 'b-', lw=2, alpha=0.8, label='Expanded pool (16 ops) — hierarchical embedding')
ax_v.set_xlabel('Re V(t)')
ax_v.set_ylabel('Im V(t)')
ax_v.set_title('Hierarchical embedding of recurrent V(t) paths')
ax_v.legend()
ax_v.grid(True, alpha=0.3)

# D(t) panels
ax_d_small.plot(t_small, D_small, 'r-', lw=1)
ax_d_small.set_title('D(t) — small pool')
ax_d_small.set_ylabel('Momentum D(t)')
ax_d_exp.plot(t_exp, D_exp, 'b-', lw=1)
ax_d_exp.set_title('D(t) — expanded pool')
ax_d_exp.set_xlabel('t')
ax_d_exp.set_ylabel('Momentum D(t)')

# Animation: live clock + V(t) trail (small pool)
hands = []
for i in range(len(small_pool)):
    hand, = ax_clock.plot([], [], 'o-', lw=2.5, markersize=7)
    hands.append(hand)
trail, = ax_v.plot([], [], 'r-', lw=3, alpha=0.95)

def animate(frame):
    t_val = t_small[frame]
    phases = (t_val * np.log(small_pool)) % (2 * np.pi)
    for j, h in enumerate(hands):
        x = np.array([0, np.cos(phases[j])])
        y = np.array([0, np.sin(phases[j])])
        h.set_data(x, y)
    trail.set_data(V_small[:frame].real, V_small[:frame].imag)
    return [*hands, trail]

ani = FuncAnimation(fig, animate, frames=len(t_small), interval=15, blit=True)

plt.tight_layout()
plt.show()

print("\nSuper-pointer states (small pool):", sp_small[:6])
print("Super-pointer states (expanded pool):", sp_exp[:6])