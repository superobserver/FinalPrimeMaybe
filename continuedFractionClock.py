from mpmath import mp, mpf, log, pi, floor, frac
mp.dps = 60

print("=== Continued-Fraction Super-Pointer Recurrence Oracle — 2-Hand Seed (7+11) ===")

p1 = mpf(7)
p2 = mpf(11)
alpha = log(p2) / log(p1)
print(f"Ratio α = log(11)/log(7) ≈ {alpha}")

# Correct continued-fraction convergents
def continued_fraction_convergents(x, max_terms=60):
    convergents = []
    x0 = x
    p_prev, q_prev = mpf(0), mpf(1)
    p, q = mpf(1), mpf(0)
    for _ in range(max_terms):
        ai = floor(x0)
        p_new = ai * p + p_prev
        q_new = ai * q + q_prev
        convergents.append((p_new, q_new))
        p_prev, q_prev = p, q
        p, q = p_new, q_new
        frac_part = x0 - ai
        if frac_part < mpf('1e-50'):
            break
        x0 = 1 / frac_part
    return convergents

convergents = continued_fraction_convergents(alpha)

# Generate candidate t from convergents
delta = pi / 12   # 30° cone
t_start = mpf('10')
candidates = []
for h, k in convergents:
    t_base = mpf('2') * pi * k / log(p1)
    frac7 = frac(t_base * log(p1) / (2 * pi))
    shift = (mpf('0.5') - frac7) * (2 * pi) / log(p1)
    t = t_base + shift
    if t < t_start:
        continue
    phase1 = frac(t * log(p1) / (2 * pi))
    phase2 = frac(t * log(p2) / (2 * pi))
    if abs(phase1 - 0.5) < mpf('1')/24 and abs(phase2 - 0.5) < mpf('1')/24:
        candidates.append(t)
    if len(candidates) >= 12:
        break

print("\nFirst 12 recurrence super-pointers (t > 10) from the 2-hand seed (7+11):")
for i, t in enumerate(candidates, 1):
    print(f"{i:2d}: t = {t}")

if len(candidates) > 1:
    gaps = [candidates[i] - candidates[i-1] for i in range(1, len(candidates))]
    print("\nGaps between consecutive alignments (saturation period of the 2-operator package):")
    for i, gap in enumerate(gaps, 1):
        print(f"Gap {i}: {gap}")