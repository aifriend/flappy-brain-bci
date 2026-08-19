"""Shared synthetic motor-imagery epochs (mirrors the simulator's ERD).

8 channels; layout matches flappy_brain.simulate: ch2/3 = left motor,
ch4 = central/Cz (feet), ch5/6 = right motor.
"""

import numpy as np

LEFT_MOTOR = (2, 3)
CENTRAL = (4,)
RIGHT_MOTOR = (5, 6)


def make_lr_epochs(n_epochs=40, fs=250.0, dur_s=1.5, seed=0):
    """leftright scheme. LEFT (0): strong mu on left motor (right-hemisphere
    ERD). RIGHT (1): strong mu on right motor (left-hemisphere ERD)."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur_s)
    t = np.arange(n) / fs
    X, y = [], []
    for i in range(n_epochs):
        label = i % 2
        ep = 0.3 * rng.standard_normal((8, n))
        mu = 2.0 * np.sin(2 * np.pi * 11 * t + rng.uniform(0, 2 * np.pi))
        strong, weak = (LEFT_MOTOR, RIGHT_MOTOR) if label == 0 else (RIGHT_MOTOR, LEFT_MOTOR)
        ep[list(strong)] += mu
        ep[list(weak)] += 0.2 * mu
        ep[list(CENTRAL)] += 0.5 * mu  # feet area active at moderate level
        X.append(ep)
        y.append(label)
    return np.array(X), np.array(y)


def make_feetrest_epochs(n_epochs=40, fs=250.0, dur_s=1.5, seed=0):
    """feetrest scheme. REST (0): mu everywhere. FEET (1): mu suppressed on
    the central (Cz) channel."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur_s)
    t = np.arange(n) / fs
    X, y = [], []
    for i in range(n_epochs):
        label = i % 2
        ep = 0.3 * rng.standard_normal((8, n))
        mu = 2.0 * np.sin(2 * np.pi * 11 * t + rng.uniform(0, 2 * np.pi))
        ep[list(LEFT_MOTOR + RIGHT_MOTOR)] += mu
        ep[list(CENTRAL)] += (0.2 if label == 1 else 1.0) * mu  # Cz ERD during FEET
        X.append(ep)
        y.append(label)
    return np.array(X), np.array(y)
