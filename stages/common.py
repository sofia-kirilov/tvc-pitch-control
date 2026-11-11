"""Shared setup for the stage scripts."""

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

DEG = math.pi / 180.0
STEP = 5 * DEG  # standard test: pitch from 0 to 5 degrees

# Baseline gains (rad of gimbal per rad of pitch error), used in stages 2-5.
# Stage 3 shows where KP and KD come from.
KP = 0.82
KI = 1.5
KD = 0.124
D_FILTER = 0.01  # s

# Tuned gains from the stage 6 sweep, used in stages 7-9.
KP_T = 1.11
KI_T = 3.1
KD_T = 0.145
SP_WEIGHT = 0.4


def tuned_pid(limit, scale=1.0):
    from rocketpid import PID

    return PID(KP_T * scale, KI_T * scale, KD_T * scale, out_limit=limit,
               d_filter_tau=D_FILTER, setpoint_weight=SP_WEIGHT)


def save(fig, name):
    fig.tight_layout()
    path = FIGURES / name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved figures/{name}")


def print_metrics(label, m):
    print(
        f"{label:<28} rise {m['rise']:.3f} s   overshoot {m['overshoot']:5.1f} %   "
        f"settle {m['settle']:.3f} s   steady-state error {m['sse'] / DEG:+.3f} deg"
    )
