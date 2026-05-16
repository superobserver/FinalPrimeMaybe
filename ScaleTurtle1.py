#!/usr/bin/env python3
"""
Exact Turtle Animation: Parametric Route of the Laplace Transform
for the Quadratic-Ribbon Sieve (Class k=11) — using the real dataset
"""

import turtle
import cmath
import sys

sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder

# ================== CONFIG ==================
WIDTH = 1800
HEIGHT = 1400
MAX_T = 3000
STEP = 1
SCALE = 12.0
SPEED = 0
MAX_EPOCHS = 25
SHOW_FINAL_TOTALS_ONLY = True      # set False to see per-comb running totals
# ===========================================

print(f"Loading exact ground-state data for class k=11 up to epoch x={MAX_EPOCHS}...")
grating_data = elder.get_grating_data(91, max_x=MAX_EPOCHS)
print(f"Loaded {len(grating_data)} exact (x, y0, p, q) tuples.")

def exact_laplace(sigma, t):
    s = complex(sigma, t)
    total = 0j
    for idx, (x_val, y, p, q) in enumerate(grating_data):
        # p-branch
        denom_p = 1 - cmath.exp(-s * p)
        if abs(denom_p) > 1e-12:
            total += cmath.exp(-s * y) / denom_p
        # q-branch
        denom_q = 1 - cmath.exp(-s * q)
        if abs(denom_q) > 1e-12:
            total += cmath.exp(-s * y) / denom_q

        if not SHOW_FINAL_TOTALS_ONLY and idx % 24 == 23:  # after each full epoch
            print(f"  after epoch {x_val:2d} | running total: {total}")

    if SHOW_FINAL_TOTALS_ONLY:
        print(f"t = {t:4d} | final Laplace value = {total}")
    return total

# ================== TURTLE SETUP ==================
turtle.setup(WIDTH, HEIGHT)
turtle.bgcolor("black")
turtle.title("Exact Laplace Route — Quadratic-Ribbon Sieve (Class 11)")

pen = turtle.Turtle()
pen.speed(SPEED)
pen.pensize(2)
pen.color("cyan")

# Draw axes
pen.color("gray")
pen.penup()
pen.goto(-WIDTH//2, 0)
pen.pendown()
pen.goto(WIDTH//2, 0)
pen.penup()
pen.goto(0, -HEIGHT//2)
pen.pendown()
pen.goto(0, HEIGHT//2)
pen.penup()

# Trace the route
pen.color("white")
pen.pendown()

sigma = 0.01
print(f"Tracing route with σ = {sigma} ...")
for ti in range(0, int(MAX_T) + 1, STEP):
    val = exact_laplace(sigma, ti)
    screen_x = val.real * SCALE
    screen_y = val.imag * SCALE
    pen.goto(screen_x, screen_y)

pen.penup()

# Labels
pen.goto(10, 10)
pen.color("red")
pen.write("Origin (0)", font=("Arial", 14, "bold"))

pen.goto(-WIDTH//2 + 40, HEIGHT//2 - 80)
pen.color("yellow")
pen.write("Exact Parametric Route of Laplace Transform\n"
          f"Re(s) = {sigma:.2f} + it    (t = 0 → {MAX_T})\n"
          "All poles lie strictly on the imaginary axis\n"
          f"Data: {len(grating_data)} real combs",
          font=("Arial", 16, "bold"))

turtle.done()