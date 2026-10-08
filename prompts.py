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

ADDED (optional intent line, re-added on top of the single-prompt design
above): build_prompt() now takes an optional `intent` string. This does
NOT bring back the old 3-call CoT design -- it's still one call, one
prompt, no generated text fed back into another call. It just lets one
extra short line of text ride along in that same single prompt when a
caller happens to have an intent description on hand (e.g. from
generate_intents.py's precomputed {sample_id: intent} map, wired in by
openemma_dataset.py). The backbone encodes whatever text is in the
prompt into hidden states regardless of where that text came from, so
this is mechanically consistent with the architecture; it's a content
change (one more sentence of context), not a structural one.

When `intent` is None or empty (the default, and what every existing
caller still gets unless it explicitly passes one), the prompt text is
byte-for-byte identical to before this change.

CAVEAT this file does not resolve on its own: if `intent` was produced
by looking at the ground-truth FUTURE trajectory (as generate_intents.py
does, deliberately, since it's an offline labeling pass), then training
with that intent in the prompt gives the model a privileged hint about
the very thing it's being asked to predict. That is only a fair,
leak-free input at inference time if whatever supplies `intent` there
is blind to the future too (e.g. a separate, image+past-only CoT call,
the way the original OpenEMMA pipeline generates intent). Passing a
future-derived intent at inference, or passing no intent at inference
after training with one, both create a train/test mismatch worth
deciding on deliberately rather than by default.
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


def build_prompt(obs, intent=None):
    """
    Builds the single shared prompt TEXT (the caller prepends
    DEFAULT_IMAGE_TOKEN + "\\n" itself -- that is a tokenizer detail, not
    a wording one, and both train_openemma_tinyvla.py's preprocess_batch
    and main.py's predict_step already do that the same way).

    obs: array-like of shape (OBS_LEN, 2) -- the vehicle's last OBS_LEN
    [speed, curvature*100] pairs, oldest first, most recent last.

    intent: optional short string (e.g. "turning left at intersection").
    When falsy (None or ""), this function behaves exactly as before --
    no intent line is added. See this module's docstring for the
    train/inference leakage caveat before passing a non-None value here.

    Deliberately small: no separate "describe the scene" / "identify
    objects" / "update intent" sub-prompts. The model must do all of
    that reasoning itself from the image, using the numeric history only
    as a supporting cue about how the vehicle has been moving.
    """
    history = format_history(obs)
    intent_line = f"Likely intent for this moment: {intent}.\n\n" if intent else ""
    return (
        "You are driving this vehicle. Using the camera image and the vehicle's own "
        f"judgement of the road ahead -- nearby vehicles, pedestrians, lane geometry, and "
        f"where the vehicle is headed -- decide how it should move for the next {FUT_LEN} "
        "timesteps (0.5s apart). Use the recent motion history below only as a hint of its "
        "current trajectory\n\n"
        f"Recent motion history, oldest to most recent ([speed m/s, curvature x100]):\n{history}\n\n"
        f"{intent_line}"
        f"Output the vehicle's predicted speed and curvature for each of the next {FUT_LEN} "
        "timesteps."
    )
