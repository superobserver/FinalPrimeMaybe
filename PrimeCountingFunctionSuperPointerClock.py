import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

print("=== Animation: Super-Pointer Correction to π(x) ===")

# Skew law t_k ≈ 1.14 * 4.96^k (k starts at 2)
c, r = 1.14, 4.96
super_pointers = [c * (r ** k) for k in range(2, 200)]
super_pointers = np.array(super_pointers)
print(f"Generated {len(super_pointers)} super-pointers (k=2 to 51)")

# Generate actual primes and π(x) from the algebraic ideal
MAX_X = 10000000
holes = []
classes = [k if k != 1 else 91 for k in elder.COPRIME_RESIDUES]
for k in classes:
    amp = elder.generate_amplitude_map(k, MAX_X//90 + 100)
    for m in range(len(amp)):
        if amp[m] == 0:
            p = 90 * m + k
            if p > 5 and p <= MAX_X:
                holes.append(p)
primes = np.sort(np.unique(holes))
print(f"Generated {len(primes)} actual primes up to {MAX_X}")

x_plot = np.logspace(1, np.log10(MAX_X), 2000)

# Smooth main term li(x) approximation
def li(x):
    return x / np.log(x) + x / (np.log(x)**2)

main_term = li(x_plot)

# Actual prime counting function
pi_actual = np.zeros_like(x_plot)
for i, xi in enumerate(x_plot):
    pi_actual[i] = np.sum(primes <= xi)

# Figure setup
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('x (log scale)')
ax.set_ylabel('Prime counting function')
ax.set_title('Prime Counting Function Updated by Super-Pointer States')
ax.grid(True, alpha=0.3)

line_main, = ax.plot(x_plot, main_term, 'b-', lw=2, label='Smooth main term li(x)')
line_actual, = ax.plot(x_plot, pi_actual, 'k-', lw=1.5, alpha=0.8, label='Actual π(x) from algebraic ideal')
line_corrected, = ax.plot([], [], 'r-', lw=2, label='Super-pointer corrected π(x)')
ax.legend()

text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

def init():
    line_corrected.set_data([], [])
    text.set_text('')
    return line_corrected, text

def animate(frame):
    # Use first (frame+2) super-pointers (k=2 to frame+2)
    current_pointers = super_pointers[:frame+1]
    correction = np.zeros_like(x_plot)
    for tn in current_pointers:
        phase = tn * np.log(x_plot)
        amp = x_plot**0.5 / tn
        correction += -amp * np.cos(phase)
    corrected = main_term + correction
    line_corrected.set_data(x_plot, corrected)
    text.set_text(f'Super-pointers used: k=2 to {frame+2}  ({len(current_pointers)} pointers)')
    return line_corrected, text

ani = FuncAnimation(fig, animate, frames=len(super_pointers), init_func=init, interval=200, blit=False, repeat=True)

ani.save('super_pointer_correction_animation.mp4', writer='ffmpeg', fps=8, dpi=200)
plt.show()

print("Animation saved as super_pointer_correction_animation.mp4")
print("The red curve updates incrementally as each new super-pointer is added.")
print("You will see the smooth blue line being progressively pulled toward the black actual prime count.")