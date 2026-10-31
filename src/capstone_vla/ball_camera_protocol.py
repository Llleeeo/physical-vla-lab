"""Opt-in formal temporal render boundary; no model/physics commands."""
import math

RESET_KEY = '/rtx-transient/post/dlss/forceParamReset'
CLAMP_KEYS = ('/rtx/reflections/sampledLighting/maxRayIntensity',
              '/rtx/directLighting/sampledLighting/maxRayIntensity')


def validated_clamps(record):
    values = record['applied']
    if set(values) != set(CLAMP_KEYS):
        raise ValueError('Unexpected training render calibration settings')
    result = {k: float(values[k]) for k in CLAMP_KEYS}
    if any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError('Invalid training render clamp')
    return result


def formal_capture(rep, quality, settings, calibration):
    """Match formal recorder reset/subframes; keep native timeline running.

    Reuse the training scene's recorded renderer clamp (not its probe ISO as
    camera exposure). No sim.render() before this boundary, no warmup physics.
    This is a protocol ablation, not a promise of pixel-identical rendering.
    """
    count = int(quality.get('temporal_subframes') or 0)
    if count != 4:
        raise ValueError('Expected balanced four-subframe profile')
    for key, value in validated_clamps(calibration).items():
        settings.set(key, value)
    settings.set(RESET_KEY, True)
    try:
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=count)
    finally:
        settings.set(RESET_KEY, False)
