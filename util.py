


import sys
import numpy as np
from scipy.spatial.distance import euclidean


# ----------------------------------------------------
# 🧩 RECTANGLE DISTANCE CALCULATION
# ----------------------------------------------------
def rect_distance(rect1, rect2):
    """
    Compute the shortest Euclidean distance between two rectangles.
    Rectangles are defined as (x1, y1, x2, y2).

    Returns:
        float: Minimum distance in pixels (0 if overlapping).
    """
    try:
        (x1, y1, x1b, y1b) = rect1
        (x2, y2, x2b, y2b) = rect2
    except Exception as e:
        print("[util] Invalid rect input:", e)
        return np.inf

    # Positional relations
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2

    # Diagonal distances
    if top and left:
        return euclidean((x1, y1b), (x2b, y2))
    elif left and bottom:
        return euclidean((x1, y1), (x2b, y2b))
    elif bottom and right:
        return euclidean((x1b, y1), (x2, y2b))
    elif right and top:
        return euclidean((x1b, y1b), (x2, y2))

    # Axis-aligned distances
    elif left:
        return float(x1 - x2b)
    elif right:
        return float(x2 - x1b)
    elif bottom:
        return float(y1 - y2b)
    elif top:
        return float(y2 - y1b)

    # Overlapping rectangles
    return 0.0


# ----------------------------------------------------
# ⚙️ PROGRESS INDICATOR (CLI)
# ----------------------------------------------------
def progress(frame_count):
    """
    Display an animated progress indicator for backend CLI use.
    Triggered when SHOW_PROCESSING_OUTPUT=False.
    """
    sys.stdout.write('\r')
    dots = "." * (1 + frame_count % 3)
    sys.stdout.write(f"Processing{dots:<3}")
    sys.stdout.flush()


# ----------------------------------------------------
# ⚡ KINETIC ENERGY CALCULATION
# ----------------------------------------------------
def kinetic_energy(point1, point2, time_step):
    """
    Calculate kinetic energy between two tracked points.

    Args:
        point1, point2: (x, y) coordinates.
        time_step: seconds between frames.

    Returns:
        int: Kinetic energy = 0.5 * v^2
    """
    try:
        speed = euclidean(point1, point2) / time_step
        return int(0.5 * speed ** 2)
    except Exception as e:
        print("[util] Kinetic energy calc error:", e)
        return 0
