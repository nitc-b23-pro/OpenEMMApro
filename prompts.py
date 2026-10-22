"""
prompts.py
Shared prompt builder for OpenEMMA, used identically at training and
inference time: one prompt built from the vehicle's recent motion
history, asking the model to predict the next OBS_LEN future steps
directly from the image and that history.
"""

OBS_LEN = 10
FUT_LEN = 10


def format_history(obs):
    """
    obs: array-like of shape (OBS_LEN, 2), rows = [speed (m/s), curvature*100],
    oldest first. Renders it as a compact numeric history block.
    """
    lines = []
    n = len(obs)
    for t in range(n):
        speed, curv = obs[t][0], obs[t][1]
        lines.append(f"t-{n - t}: speed={float(speed):.2f} m/s, curvature={float(curv):.2f}")
    return "\n".join(lines)


def build_prompt(obs):
    """Builds the prompt text (the caller adds any image placeholder the backbone needs)."""
    history = format_history(obs)
    return (
        "You are driving this vehicle. Using the camera image and the vehicle's own "
        f"judgement of the road ahead -- nearby vehicles, pedestrians, lane geometry, and "
        f"where the vehicle is headed -- decide how it should move for the next {FUT_LEN} "
        "timesteps (0.5s apart). Use the recent motion history below only as a hint of its "
        "current trajectory\n\n"
        f"Recent motion history, oldest to most recent ([speed m/s, curvature x100]):\n{history}\n\n"
        f"Output the vehicle's predicted speed and curvature for each of the next {FUT_LEN} "
        "timesteps."
    )
