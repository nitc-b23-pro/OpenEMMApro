"""
route_intent.py
Pure, deterministic, map-free "turn-by-turn navigator" intent for a
window of a nuScenes scene -- no image, no VLM, no GPU. It classifies
intent purely from the ego vehicle's own recorded path geometry
(curvature + speed over time), the same way a Google-Maps-style
navigator announces "next left turn" or "continue straight" from the
planned ROUTE, not from what's happening in the scene around the car
(no critical objects, no "slow down for pedestrian" reasoning -- that is
exactly what this module does NOT do, by design).

Vocabulary (atomic labels; a window's final phrase is these, possibly
combined -- see build_phrase):
    "continue straight"
    "bear left" / "bear right"     (a gentle, sustained curve -- e.g. a highway bend)
    "left turn" / "right turn"     (a real turn -- e.g. at an intersection)
    "u-turn"                       (a near-total heading reversal)

How a label is chosen
----------------------
Heading change over a segment is computed the same way the rest of this
project already integrates curvature into heading (see
utils.IntegrateCurvatureForPoints: heading += curvature * speed * dt),
summed across the segment and converted to degrees. The SIGN convention
matches that function too: positive curvature/heading-change = turning
left (counter-clockwise), negative = turning right -- this is not a new
convention, it's the one already used throughout main.py/utils.py.

A window can span more than one SEGMENT (segment_steps long each, e.g.
10 steps = 5s). Each segment gets its own atomic label from its own
cumulative heading change; consecutive identical labels collapse, and
the surviving labels are joined into one short phrase (build_phrase),
which is how "combinations of these simple intents" (e.g. "continue
straight, then left turn") come about.

READ THIS before picking `start_idx` in a real pipeline
---------------------------------------------------------
This module does not know or care what a "prediction window" is -- it
just classifies whatever (curv, speed) slice you give it, starting at
whatever `start_idx` you pass in. That choice matters a lot:

- If you start the look-ahead at the SAME index where the model's
  future-prediction target begins (i.e. describe the very 10 steps the
  diffusion head must output), the resulting label is a coarse summary
  of the model's own answer -- a much weaker hint than handing over the
  literal (speed, curvature) numbers, but still derived from the target
  window itself, not from an independently-known route.
- If you start the look-ahead AFTER the prediction window ends (e.g. at
  i + TTL_LEN instead of i + OBS_LEN, in openemma_dataset.py's indexing),
  the label describes the route beyond what's being predicted right
  now -- much closer to what a real turn-by-turn navigator actually
  supplies (it knows the route further out than your next 5 seconds),
  and it does not directly describe the target window.

generate_intents.py defaults to the second, safer option
(LOOKAHEAD_OFFSET = FUT_LEN) -- see its own docstring and CLI flag if you
want to change this deliberately.
"""
import numpy as np

DT = 0.5   # nuScenes keyframes are 2 Hz -- matches every other DT in this project

BEAR_DEG = 12.0     # |heading change| at/above this, below TURN_DEG -> "bear left/right"
TURN_DEG = 45.0     # |heading change| at/above this, below UTURN_DEG -> "left turn"/"right turn"
UTURN_DEG = 150.0   # |heading change| at/above this -> "u-turn"


def segment_heading_change_deg(curv_segment, speed_segment, dt=DT):
    """
    Sum of curvature*speed*dt over a segment, in degrees. Same formula as
    utils.IntegrateCurvatureForPoints's heading update, just summed
    directly instead of accumulated point-by-point (we only need the
    total turn over the segment, not the intermediate positions).
    """
    curv_segment = np.asarray(curv_segment, dtype=float)
    speed_segment = np.asarray(speed_segment, dtype=float)
    delta_rad = float(np.sum(curv_segment * speed_segment * dt))
    return np.degrees(delta_rad)


def label_for_delta(delta_deg):
    mag = abs(delta_deg)
    if mag >= UTURN_DEG:
        return "u-turn"
    if mag >= TURN_DEG:
        return "left turn" if delta_deg > 0 else "right turn"
    if mag >= BEAR_DEG:
        return "bear left" if delta_deg > 0 else "bear right"
    return "continue straight"


def segment_labels(curv, speed, start_idx, lookahead_steps, segment_steps, dt=DT):
    """
    Splits curv[start_idx : start_idx+lookahead_steps] (and the matching
    speed slice) into consecutive chunks of segment_steps, clipped to
    whatever is actually available (the scene may end before a full
    look-ahead window is reached), and returns one label per chunk.
    Returns [] if start_idx is already at or past the end of the arrays
    (e.g. a window near the very end of a scene).
    """
    n = len(curv)
    end_idx = min(start_idx + lookahead_steps, n)
    labels = []
    for chunk_start in range(start_idx, end_idx, segment_steps):
        chunk_end = min(chunk_start + segment_steps, end_idx)
        if chunk_end <= chunk_start:
            break
        delta_deg = segment_heading_change_deg(curv[chunk_start:chunk_end], speed[chunk_start:chunk_end], dt)
        labels.append(label_for_delta(delta_deg))
    return labels


def build_phrase(labels, max_segments=2):
    """
    Collapses consecutive duplicate labels, keeps at most max_segments of
    the result (keeping the nearest-term ones), and renders "left turn"/
    "right turn" as "next left turn"/"next right turn" only when it's the
    very first thing in the phrase (matching how a navigator announces
    the upcoming turn) -- a turn mentioned later in a combined phrase is
    rendered without "next" so it doesn't read as "next ... then next ...".
    """
    if not labels:
        return "continue straight"   # no look-ahead data available (end of scene) -- the neutral default

    collapsed = []
    for lab in labels:
        if not collapsed or collapsed[-1] != lab:
            collapsed.append(lab)
    collapsed = collapsed[:max_segments]

    parts = []
    for i, lab in enumerate(collapsed):
        if i == 0 and lab in ("left turn", "right turn"):
            parts.append(f"next {lab}")
        else:
            parts.append(lab)
    return ", then ".join(parts)


def route_intent_for_window(curv, speed, start_idx, lookahead_steps=20, segment_steps=10, dt=DT, max_segments=2):
    """
    One-call convenience: segment_labels() + build_phrase(). curv/speed
    are the FULL per-scene arrays (not just a 10-step obs/fut slice) --
    see this module's docstring for why start_idx's choice relative to a
    prediction window matters.
    """
    labels = segment_labels(curv, speed, start_idx, lookahead_steps, segment_steps, dt)
    return build_phrase(labels, max_segments=max_segments)
