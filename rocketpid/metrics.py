"""Numbers that describe a step response."""

import numpy as np


def step_metrics(t, y, target, start=0.0, band=0.02):
    """Rise time (10-90%), overshoot (%), settling time (within band), steady-state error."""
    size = target - start
    if np.any(np.isnan(y)) or size == 0:
        return dict(rise=np.nan, overshoot=np.nan, settle=np.nan, sse=np.nan)

    frac = (y - start) / size
    above10 = np.argmax(frac >= 0.1) if np.any(frac >= 0.1) else None
    above90 = np.argmax(frac >= 0.9) if np.any(frac >= 0.9) else None
    rise = t[above90] - t[above10] if above10 is not None and above90 is not None else np.nan

    overshoot = max(0.0, (frac.max() - 1.0) * 100.0)

    outside = np.abs(frac - 1.0) > band
    if not outside.any():
        settle = t[0]
    elif outside[-1]:
        settle = np.nan
    else:
        settle = t[np.nonzero(outside)[0][-1] + 1]

    tail = max(1, len(y) // 20)
    sse = target - y[-tail:].mean()
    return dict(rise=rise, overshoot=overshoot, settle=settle, sse=sse)


def itae(t, y, setpoint):
    """Integral of time * |error|. Lower is better; late error costs more."""
    if np.any(np.isnan(y)):
        return np.inf
    dt = t[1] - t[0]
    return float(np.sum(t * np.abs(setpoint - y)) * dt)
