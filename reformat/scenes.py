"""
Scene detection, chunking, fps normalisation and canvas math.

These are the defaults antilopax.ai's ReFormat tool runs with. They were
tuned on real uploads, so change them only for a reason.
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger("ReFormat")

# Production defaults (what the hosted tool sends to its workers).
DETECT_THRESHOLD = 38.0     # PySceneDetect ContentDetector threshold
MIN_SCENE_SECONDS = 1.5     # shorter detections are merged into a neighbour
MAX_CHUNK_SECONDS = 10.0    # LTX-2.3 at 1080p two-pass OOMs a 32 GB GPU above ~10 s
MAX_FPS = 30.0              # clips are halved until <= 30 fps (frame count drives VRAM and time)
SEAM_FRAMES = 2             # dissolve of the outpainted fill across an auto-split seam; short because the
                            # continuation is anchored on the previous chunk's last frame in both passes
TARGET_EDGE = 1920          # long edge of the preset canvas (1920x1088 / 1088x1920)

DIRECTIONS = ("landscape-to-portrait", "portrait-to-landscape")


def detect_scenes(video_path: str, threshold: float = DETECT_THRESHOLD,
                  min_scene_length: float = MIN_SCENE_SECONDS) -> list[dict]:
    """Scene boundaries via PySceneDetect's ContentDetector, short scenes merged into neighbours."""
    from scenedetect import ContentDetector, detect  # imported here so the pack loads without it

    scene_list = detect(video_path, ContentDetector(threshold=threshold))
    scenes = []
    for i, (start, end) in enumerate(scene_list):
        scenes.append({
            "index": i,
            "start_time": start.get_seconds(),
            "end_time": end.get_seconds(),
            "start_timecode": str(start),
            "end_timecode": str(end),
        })

    if min_scene_length > 0 and len(scenes) > 1:
        merged = [scenes[0]]
        for scene in scenes[1:]:
            prev = merged[-1]
            prev_duration = prev["end_time"] - prev["start_time"]
            curr_duration = scene["end_time"] - scene["start_time"]
            if prev_duration < min_scene_length:
                merged[-1]["end_time"] = scene["end_time"]
                merged[-1]["end_timecode"] = scene["end_timecode"]
                log.info("merged short scene (%.1fs) with the next one", prev_duration)
            elif curr_duration < min_scene_length and scene is scenes[-1]:
                merged[-1]["end_time"] = scene["end_time"]
                merged[-1]["end_timecode"] = scene["end_timecode"]
                log.info("merged short final scene (%.1fs) with the previous one", curr_duration)
            else:
                merged.append(scene)
        for i, scene in enumerate(merged):
            scene["index"] = i
        if len(merged) < len(scenes):
            log.info("merged %d short scenes -> %d scenes", len(scenes) - len(merged), len(merged))
        scenes = merged
    return scenes


def subdivide_long_scenes(scenes: list[dict], max_duration: float = MAX_CHUNK_SECONDS) -> list[dict]:
    """Cut every scene longer than max_duration into equal chunks <= max_duration.

    Chunks of one shot share a `group` (the scene's index before chunking) and
    carry auto_split=True; that is what continuation and seam blending key on.
    """
    result = []
    for scene in scenes:
        duration = scene["end_time"] - scene["start_time"]
        group = scene["index"]
        if duration <= max_duration:
            scene = dict(scene, auto_split=False, group=group, chunk=0, chunk_count=1)
            result.append(scene)
            continue
        num_chunks = int(duration / max_duration) + (1 if duration % max_duration > 0 else 0)
        chunk_duration = duration / num_chunks
        log.info("subdividing scene %d (%.1fs) into %d chunks of ~%.1fs", group, duration, num_chunks, chunk_duration)
        for i in range(num_chunks):
            chunk_start = scene["start_time"] + i * chunk_duration
            chunk_end = min(scene["start_time"] + (i + 1) * chunk_duration, scene["end_time"])
            result.append({
                "index": 0,
                "start_time": chunk_start,
                "end_time": chunk_end,
                "start_timecode": f"{chunk_start:.3f}",
                "end_timecode": f"{chunk_end:.3f}",
                "auto_split": True,
                "group": group,
                "chunk": i,
                "chunk_count": num_chunks,
            })
    for i, scene in enumerate(result):
        scene["index"] = i
    return result


def normalized_fps(src_fps: float, max_fps: float = MAX_FPS) -> float:
    """Halve the frame rate until it is <= max_fps; never upsample.

    VRAM and generation time scale with the FRAME COUNT, not the seconds: a
    10 s clip at 60 fps is 600 frames and OOMs a 32 GB GPU, the same 10 s at
    30 fps fits. Halving keeps the decimation exact (59.94 -> 29.97, no
    judder), and low-fps sources are left alone.
    """
    fps = float(src_fps or max_fps)
    while fps > max_fps + 0.5:
        fps /= 2
    return round(fps, 3)


def _js_round(v: float) -> int:
    """JavaScript's Math.round (half up), to match the hosted tool's canvas math exactly."""
    return int(math.floor(v + 0.5))


def snap32(v: float) -> int:
    return _js_round(v / 32) * 32


def canvas_layout(src_w: int, src_h: int, direction: str, split_percent: int = 50,
                  target_edge: int = TARGET_EDGE) -> dict:
    """Canvas size, padding and slot rect for a source clip of src_w x src_h.

    The source band spans the canvas's short axis (scaled, snapped to 32) and
    sits at split_percent of the padding from the top (landscape-to-portrait)
    or from the left (portrait-to-landscape). Same math as the hosted tool.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    sp = max(0, min(100, int(split_percent)))
    clamped = False
    if direction == "landscape-to-portrait":
        ch = snap32(target_edge)
        cw = snap32(target_edge * 9 / 16)
        scale = cw / src_w
        scaled_h = snap32(_js_round(src_h * scale))
        if scaled_h > ch:
            scaled_h, clamped = ch, True
        pad_total = max(0, ch - scaled_h)
        pt = _js_round(pad_total * sp / 100)
        pb = pad_total - pt
        pl = pr = 0
    else:
        cw = snap32(target_edge)
        ch = snap32(target_edge * 9 / 16)
        scale = ch / src_h
        scaled_w = snap32(_js_round(src_w * scale))
        if scaled_w > cw:
            scaled_w, clamped = cw, True
        pad_total = max(0, cw - scaled_w)
        pl = _js_round(pad_total * sp / 100)
        pr = pad_total - pl
        pt = pb = 0
    slot_w = cw - pl - pr
    slot_h = ch - pt - pb
    return {
        "canvas_width": cw,
        "canvas_height": ch,
        "pad_top": pt,
        "pad_bottom": pb,
        "pad_left": pl,
        "pad_right": pr,
        "slot": {"x": pl, "y": pt, "w": slot_w, "h": slot_h},
        "clamped": clamped,
    }
