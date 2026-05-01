#!/usr/bin/env python3
"""
Exact Turtle Animation: Parametric Route of the Laplace Transform
for the Quadratic-Ribbon Sieve (Class k=11) — using the real dataset
"""

import turtle
import cmath
import time
import sys

# Add module path (adjust if your file is named differently)
sys.path.append('/home/workdir/attachments')
import April1Sieve2 as elder   # <-- your module

# ================== CONFIG ==================
WIDTH = 1800
HEIGHT = 1400
MAX_T = 1500                  # larger range for richer path
STEP = 1                     # sampling step (trade-off speed vs smoothness)
SCALE = 84.5                   # zoom factor
SPEED = 0                     # turtle speed (0 = fastest)
MAX_EPOCHS = 900               # how many ribbon levels (x) to include
# ===========================================

# Load the exact grating data ONCE (all generators, all epochs)
print("Loading exact ground-state data for class k=11...")
grating_data = elder.get_grating_data(83, max_x=MAX_EPOCHS)
print(f"Loaded {len(grating_data)} exact (x, y0, p, q) tuples.")

def exact_laplace(sigma, t):
    """Exact Laplace transform using the real y0, p, q from the module."""
    s = complex(sigma, t)
    total = 0j
    for x, y, p, q in grating_data:
        # One comb for p-branch
        denom_p = 1 - cmath.exp(-s * p)
        if abs(denom_p) > 1e-12:
            total += cmath.exp(-s * y) / denom_p
        # One comb for q-branch
        denom_q = 1 - cmath.exp(-s * q)
        if abs(denom_q) > 1e-12:
            total += cmath.exp(-s * y) / denom_q
            print("this is total", total)
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
pen.goto(WIDTH//2, 0)      # Real axis
pen.penup()
pen.goto(0, -HEIGHT//2)
pen.pendown()
pen.goto(0, HEIGHT//2)     # Imaginary axis
pen.penup()

# Trace the exact route
pen.color("white")
pen.pendown()

sigma = 0.05
for ti in range(0, int(MAX_T) + 1, STEP):
    val = exact_laplace(sigma, ti)
    screen_y = val.real * SCALE
    screen_x = val.imag * SCALE
    pen.goto(screen_x, screen_y)
    #time.sleep(0.008)          # gentle animation

pen.penup()

# Label origin
pen.goto(10, 10)
pen.color("red")
pen.write("Origin (0)", font=("Arial", 14, "bold"))

# Final annotation
pen.goto(-WIDTH//2 + 40, HEIGHT//2 - 80)
pen.color("yellow")
pen.write("Exact Parametric Route of Laplace Transform\n"
          "Re(s) = 0.01, Im(s) = t\n"
          "All poles lie strictly on the imaginary axis\n"
          f"Data: {len(grating_data)} real combs from elder_sieve.py",
          font=("Arial", 16, "bold"))

turtle.done()