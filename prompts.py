"""
prompts.py
The ONE shared prompt used by BOTH training (openemma_dataset.py, used
inside train_openemma_tinyvla.py) and inference (main.py) for
OpenEMMA-TinyVLA.

Why this file exists
---------------------
The original OpenEMMA/EMMA pipeline made 3 separate text calls before the
final motion call: "describe the scene", "describe critical objects",
"update the intent" -- and fed all of that generated text back in as a
huge prompt for a 4th call. That design assumes a text-generation head
that can usefully produce and consume those intermediate strings.

Here the final output does NOT come from a text head at all -- it comes
from a diffusion policy head (see build_model.py / droid_unet_diffusion.py)
that conditions on a single pooled hidden state of the whole prompt
sequence (image + text, globally average-pooled). There is nothing for a
separate scene/object/intent TEXT sub-step to hand off to downstream,
since the diffusion head never reads generated text -- it only reads
hidden states of whatever is in the prompt. So instead of asking the
model to narrate its reasoning across several calls, this single prompt
asks it to internalize that reasoning (look at the scene, judge nearby
agents' behavior, infer intent) and go straight to the numeric decision,
using the image and the raw motion history as its only inputs.

Both training and inference build this EXACT same text for a given
history. The only difference between the two is what else is passed to
model(...): training also passes `actions=` (10 future [speed, curvature]
GT pairs) as the diffusion regression target; inference does not (it
runs the denoising loop instead, via eval=True).
"""

OBS_LEN = 10
FUT_LEN = 10


def format_history(obs):
    """
    obs: array-like of shape (OBS_LEN, 2). Row t = [speed (m/s), curvature*100]
    for one of the last OBS_LEN timesteps (0.5s apart, oldest first).
    Renders it as a compact numeric history block for the prompt text.
    """
    lines = []
    n = len(obs)
    for t in range(n):
        speed, curv = obs[t][0], obs[t][1]
        steps_ago = n - t
        lines.append(f"t-{steps_ago}: speed={float(speed):.2f} m/s, curvature={float(curv):.2f}")
    return "\n".join(lines)


def build_prompt(obs):
    """
    Builds the single shared prompt TEXT (the caller prepends
    DEFAULT_IMAGE_TOKEN + "\\n" itself -- that is a tokenizer detail, not
    a wording one, and both train_openemma_tinyvla.py's preprocess_batch
    and main.py's predict_step already do that the same way).

    obs: array-like of shape (OBS_LEN, 2) -- the vehicle's last OBS_LEN
    [speed, curvature*100] pairs, oldest first, most recent last.

    Deliberately small: no separate "describe the scene" / "identify
    objects" / "update intent" sub-prompts. The model must do all of
    that reasoning itself from the image, using the numeric history only
    as a supporting cue about how the vehicle has been moving.
    """
    history = format_history(obs)
    return (
        "You are driving this vehicle. Using the camera image and the vehicle's own "
        f"judgement of the road ahead -- nearby vehicles, pedestrians, lane geometry, and "
        f"where the vehicle is headed -- decide how it should move for the next {FUT_LEN} "
        "timesteps (0.5s apart). Use the recent motion history below only as a hint of its "
        "current trajectory, not as something to copy forward.\n\n"
        f"Recent motion history, oldest to most recent ([speed m/s, curvature x100]):\n{history}\n\n"
        f"Output the vehicle's predicted speed and curvature for each of the next {FUT_LEN} "
        "timesteps."
    )
