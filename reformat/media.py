"""
ffmpeg / ffprobe helpers for the ReFormat node pack.

Lifted from the CPU worker that runs behind antilopax.ai's ReFormat tool
(scene split, seam blend, stitch), minus the storage and billing parts.
Everything here runs on the CPU and shells out to ffmpeg / ffprobe.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import warnings

import numpy as np
import torch

log = logging.getLogger("ReFormat")

FFMPEG = os.environ.get("REFORMAT_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("REFORMAT_FFPROBE", "ffprobe")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpg", ".mpeg", ".wmv", ".ts")


def require_ffmpeg() -> None:
    missing = [t for t in (FFMPEG, FFPROBE) if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not found on PATH. Install ffmpeg (https://ffmpeg.org/download.html) "
            "or point REFORMAT_FFMPEG / REFORMAT_FFPROBE at the binaries."
        )


def run(cmd: list[str], what: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command and raise a readable error with ffmpeg's last lines on failure."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or "").strip()[-600:]
        raise RuntimeError(f"{what} failed (exit {r.returncode}): {tail}")
    return r


def _ffpath(p: str) -> str:
    """ffmpeg is happier with forward slashes on Windows, and so are concat lists."""
    return p.replace("\\", "/")


_PASSTHROUGH: list[str] | None = None


def passthrough_args() -> list[str]:
    """Output options that make ffmpeg hand back exactly the stored frames.

    Without them the constant-frame-rate muxing fills up to the container
    duration, which runs a few milliseconds past the last frame whenever the
    clip carries AAC audio, and one duplicated frame appears at the end.
    `-fps_mode` is ffmpeg 5.1+, `-vsync` the older spelling (Ubuntu 22.04 ships 4.4).
    """
    global _PASSTHROUGH
    if _PASSTHROUGH is None:
        r = subprocess.run(
            [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "nullsrc=s=16x16:r=5:d=0.2",
             "-fps_mode", "passthrough", "-f", "null", "-"],
            capture_output=True,
        )
        _PASSTHROUGH = ["-fps_mode", "passthrough"] if r.returncode == 0 else ["-vsync", "passthrough"]
    return _PASSTHROUGH


# ─── probing ───────────────────────────────────────────────────────────────────

def get_video_info(path: str) -> dict:
    """Duration, fps, display size (rotation-aware), frame count, audio flag."""
    r = run(
        [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        "ffprobe",
    )
    data = json.loads(r.stdout or "{}")

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    if not video_stream:
        raise ValueError(f"No video stream found in {path}")

    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    duration = float(data.get("format", {}).get("duration") or video_stream.get("duration") or 0.0)
    try:
        video_duration = float(video_stream.get("duration") or duration)
    except ValueError:
        video_duration = duration
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # Rotation metadata (phone recordings): swap to display dimensions.
    rotation = 0
    for sd in video_stream.get("side_data_list", []):
        if "rotation" in sd:
            rotation = abs(int(sd["rotation"]))
            break
    if rotation == 0:
        rotation = abs(int(video_stream.get("tags", {}).get("rotate", "0") or 0))
    is_rotated = rotation in (90, 270)
    display_width = height if is_rotated else width
    display_height = width if is_rotated else height

    frame_count = int(round(duration * fps))

    avg_str = str(video_stream.get("avg_frame_rate") or "")
    fps_avg = 0.0
    try:
        an, ad = (avg_str.split("/") + ["1"])[:2]
        fps_avg = float(an) / float(ad) if float(ad) else 0.0
    except (ValueError, ZeroDivisionError):
        fps_avg = 0.0
    if not fps_avg or fps_avg <= 0 or fps_avg > 240:
        fps_avg, avg_str = fps, video_stream.get("r_frame_rate", "30/1")
    try:
        nb_frames = int(video_stream.get("nb_frames") or 0) or None
    except ValueError:
        nb_frames = None

    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))

    if display_width > display_height:
        orientation = "landscape"
    elif display_height > display_width:
        orientation = "portrait"
    else:
        orientation = "square"

    return {
        "duration": duration,
        "fps": round(fps, 3),
        "width": display_width,
        "height": display_height,
        "stored_width": width,
        "stored_height": height,
        "rotation": rotation,
        "is_rotated": is_rotated,
        "orientation": orientation,
        "frame_count": frame_count,
        "has_audio": has_audio,
        "fps_avg": round(fps_avg, 3),
        "fps_avg_str": avg_str,
        "nb_frames": nb_frames,
        "video_duration": video_duration,
    }


def count_frames(path: str) -> int:
    """Exact video frame count (nb_frames is reliable for ffmpeg-written mp4s); 0 when unknown."""
    try:
        r = run(
            [FFPROBE, "-v", "quiet", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
             "-of", "csv=p=0", path],
            "ffprobe nb_frames",
        )
        return int((r.stdout or "0").strip() or 0)
    except Exception:
        return 0


# ─── scaling ───────────────────────────────────────────────────────────────────

def write_detection_proxy(input_path: str, output_path: str, info: dict, direction: str) -> tuple[int, int]:
    """DETECTION PROXY ONLY (720 edge). Never cut delivery clips from this.

    PySceneDetect derives its internal downscale factor from the input
    resolution, so the cut points move when this resolution changes. The proxy
    stays frozen at a 720 edge so a given video always splits the same way.
    """
    w, h, fps = info["width"], info["height"], info["fps"]
    if direction == "landscape-to-portrait":
        target_w = 720
        target_h = int(round(h * (720 / w)))
    elif direction == "portrait-to-landscape":
        target_h = 720
        target_w = int(round(w * (720 / h)))
    else:
        if w > h:
            target_w = 1280
            target_h = int(round(h * (1280 / w)))
        else:
            target_h = 1280
            target_w = int(round(w * (1280 / h)))
    target_w += target_w % 2
    target_h += target_h % 2
    run(
        [FFMPEG, "-y", "-v", "error", "-i", input_path,
         "-vf", f"scale={target_w}:{target_h}", "-r", str(fps),
         "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-metadata:s:v", "rotate=0", output_path],
        "detection proxy encode",
    )
    return target_w, target_h


def scale_for_delivery(input_path: str, output_path: str, info: dict, direction: str) -> tuple[str, int, int]:
    """Scale the source once to what the outpaint canvas consumes, never upscaling.

    The preset canvas is 1920x1088 / 1088x1920. The source band spans the
    canvas's short axis, so landscape-to-portrait scales the width to 1088,
    portrait-to-landscape the height to 1088, anything else the long edge to
    1920. A source already at or below the target passes through untouched
    (the returned path is then the input path).
    """
    w, h, fps = info["width"], info["height"], info["fps"]
    if direction == "landscape-to-portrait":
        scale = 1088 / w
    elif direction == "portrait-to-landscape":
        scale = 1088 / h
    else:
        scale = 1920 / max(w, h)
    if scale >= 1.0:
        return input_path, w, h

    target_w = int(round(w * scale))
    target_h = int(round(h * scale))
    target_w += target_w % 2
    target_h += target_h % 2
    cmd = [FFMPEG, "-y", "-v", "error", "-i", input_path,
           "-vf", f"scale={target_w}:{target_h}", "-r", str(fps),
           "-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    if info.get("has_audio"):
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-metadata:s:v", "rotate=0", output_path]
    run(cmd, "delivery scale")
    return output_path, target_w, target_h


# ─── cutting ───────────────────────────────────────────────────────────────────

def split_scene(video_path: str, start: float, end: float, output_path: str, fps: float | None = None) -> None:
    """Extract one clip, re-encoded at the normalized fps."""
    cmd = [FFMPEG, "-y", "-v", "error", "-i", video_path,
           "-ss", str(start), "-to", str(end),
           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-c:a", "aac", "-b:a", "192k",
           "-avoid_negative_ts", "make_zero"]
    if fps:
        cmd += ["-r", str(fps)]
    cmd.append(output_path)
    run(cmd, "scene split")


def extract_audio(video_path: str, output_path: str) -> None:
    run([FFMPEG, "-y", "-v", "error", "-i", video_path, "-vn", "-c:a", "aac", "-b:a", "192k", output_path],
        "audio extract")


def extract_thumbnail(video_path: str, timestamp: float, output_path: str, max_width: int = 768) -> None:
    """One JPEG frame at `timestamp` (clamped into the video), first frame as fallback."""
    try:
        r = run([FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", video_path], "ffprobe")
        duration = float(json.loads(r.stdout or "{}").get("format", {}).get("duration", 999999))
        if timestamp >= duration:
            timestamp = max(0.0, duration - 0.1)
    except Exception:
        pass
    vf = f"scale='min({max_width},iw)':-2"
    try:
        run([FFMPEG, "-y", "-v", "error", "-ss", str(timestamp), "-i", video_path,
             "-frames:v", "1", "-vf", vf, "-f", "image2", "-q:v", "2", output_path], "thumbnail")
    except RuntimeError:
        run([FFMPEG, "-y", "-v", "error", "-i", video_path,
             "-frames:v", "1", "-vf", vf, "-f", "image2", "-q:v", "2", output_path], "thumbnail (first frame)")


# ─── frames <-> tensors ────────────────────────────────────────────────────────

def _raw_to_tensor(buf: bytes, w: int, h: int) -> torch.Tensor:
    frame_bytes = w * h * 3
    n = len(buf) // frame_bytes
    if n == 0:
        raise RuntimeError("ffmpeg produced no frames")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch warns that bytes are read-only; .float() copies anyway
        t = torch.frombuffer(buf, dtype=torch.uint8, count=n * frame_bytes)
    return t.reshape(n, h, w, 3).float().div_(255.0)


def decode_frames(path: str) -> tuple[torch.Tensor, dict]:
    """All frames of a clip as a ComfyUI IMAGE batch [N, H, W, 3] float32 in 0..1."""
    info = get_video_info(path)
    w, h = info["width"], info["height"]
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path, *passthrough_args(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"frame decode failed (exit {r.returncode}): {r.stderr.decode(errors='replace')[-600:]}")
    return _raw_to_tensor(r.stdout, w, h), info


def last_frame(path: str) -> torch.Tensor:
    """The exact last frame of a clip as an IMAGE batch of one."""
    info = get_video_info(path)
    w, h = info["width"], info["height"]
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-sseof", "-0.5", "-i", path, *passthrough_args(),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    buf = r.stdout if r.returncode == 0 else b""
    if len(buf) < w * h * 3:
        frames, _ = decode_frames(path)
        return frames[-1:]
    return _raw_to_tensor(buf, w, h)[-1:]


def write_png(frame: torch.Tensor, path: str) -> None:
    """One IMAGE frame [H, W, 3] (0..1) to PNG via ffmpeg, no PIL needed."""
    h, w = frame.shape[0], frame.shape[1]
    data = frame.clamp(0, 1).mul(255).round().to(torch.uint8).contiguous().cpu().numpy().tobytes()
    r = subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-i", "-", "-frames:v", "1", path],
        input=data, capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"png write failed: {r.stderr.decode(errors='replace')[-400:]}")


def encode_frames(images: torch.Tensor, path: str, fps: float, crf: int = 19) -> None:
    """IMAGE batch -> h264 mp4 (yuv420p, no audio), the same codec settings the production graph saves with."""
    n, h, w = images.shape[0], images.shape[1], images.shape[2]
    even_w, even_h = w - (w % 2), h - (h % 2)
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "fast", "-crf", str(int(crf)), "-pix_fmt", "yuv420p"]
    if (even_w, even_h) != (w, h):
        cmd += ["-vf", f"crop={even_w}:{even_h}:0:0"]
    cmd += ["-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        step = 16
        for i in range(0, n, step):
            chunk = images[i:i + step].clamp(0, 1).mul(255).round().to(torch.uint8).contiguous().cpu().numpy()
            proc.stdin.write(chunk.tobytes())
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"chunk encode failed (exit {rc}): {err[-600:]}")


# ─── stitching ─────────────────────────────────────────────────────────────────

def _concat_entry(p: str) -> str:
    return "file '" + _ffpath(os.path.abspath(p)).replace("'", "'\\''") + "'\n"


def stitch_clips(clip_paths: list[str], audio_path: str | None, output_path: str) -> bool:
    """Concatenate clips and lay the original audio over the result. Returns True when audio was used."""
    concat_file = output_path + ".concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for clip in clip_paths:
            f.write(_concat_entry(clip))

    valid_audio = False
    if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 100:
        try:
            r = subprocess.run([FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
                               capture_output=True, text=True, timeout=10)
            valid_audio = "format" in (json.loads(r.stdout) if r.stdout else {})
        except Exception as e:
            log.warning("audio probe failed (%s), stitching without audio", e)

    try:
        if valid_audio:
            r = subprocess.run(
                [FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat_file, "-i", audio_path,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-c:a", "aac", "-b:a", "192k",
                 "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-movflags", "+faststart", output_path],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                return True
            log.warning("stitch with audio failed (exit %s): %s; retrying without audio",
                        r.returncode, (r.stderr or "")[-400:])
        run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat_file,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", "-movflags", "+faststart", output_path],
            "stitch")
        return False
    finally:
        try:
            os.remove(concat_file)
        except OSError:
            pass


def blend_seams(clip_paths: list[str], seams: list[dict], tmpdir: str) -> int:
    """Soften auto-split seams: crossfade the outpainted fill across the seam.

    For each seam between clip A (index after_clip) and clip B (after_clip+1),
    B's first `frames` frames are blended from A's last frame toward B, but
    only in the outpainted region: the source slot rect is composited back
    hard-cut for the whole clip, because the source footage is continuous
    across an auto-split and must not be ghosted. Replaces B's entry in
    clip_paths with the blended file. Best-effort: a failing seam logs a
    warning and keeps the plain hard cut. Returns the number of blended seams.
    """
    done = 0
    for si, seam in enumerate(seams):
        try:
            idx = int(seam["after_clip"])
            slot = seam["slot"]
            n_frames = max(2, min(int(seam.get("frames", 8)), 30))
            if idx < 0 or idx + 1 >= len(clip_paths):
                log.warning("seam %d: after_clip %d out of range, skipping", si, idx)
                continue

            clip_a, clip_b = clip_paths[idx], clip_paths[idx + 1]
            info_a = get_video_info(clip_a)
            info_b = get_video_info(clip_b)
            w, h = info_b["stored_width"], info_b["stored_height"]
            if (info_a["stored_width"], info_a["stored_height"]) != (w, h):
                log.warning("seam %d: canvas mismatch %dx%d vs %dx%d, skipping",
                            si, info_a["stored_width"], info_a["stored_height"], w, h)
                continue

            # Clamp the slot rect to the canvas; even values for 4:2:0 chroma.
            sx = max(0, min(int(slot["x"]), w - 2)) & ~1
            sy = max(0, min(int(slot["y"]), h - 2)) & ~1
            sw = max(2, min(int(slot["w"]), w - sx)) & ~1
            sh = max(2, min(int(slot["h"]), h - sy)) & ~1
            fps = info_b["fps"] or 24.0

            # A's last frame = the state the fill must morph away from.
            last_png = os.path.join(tmpdir, f"seam_{si:02d}_last.png")
            write_png(last_frame(clip_a)[0], last_png)

            # Ramp reaches full B at frame n_frames-1; the expression is pure B
            # afterwards, so everything past the ramp re-encodes unchanged.
            expr = f"A*(1-min((N+1)/{n_frames},1))+B*min((N+1)/{n_frames},1)"
            filtergraph = (
                f"[0:v]scale={w}:{h},format=yuv420p,fps={fps}[still];"
                f"[1:v]format=yuv420p[base];"
                f"[still][base]blend=all_expr='{expr}':shortest=1[blended];"
                f"[1:v]crop={sw}:{sh}:{sx}:{sy}[src];"
                f"[blended][src]overlay={sx}:{sy}[out]"
            )
            blended = os.path.join(tmpdir, f"seam_{si:02d}_blended.mp4")
            r = subprocess.run(
                [FFMPEG, "-y", "-v", "error",
                 "-loop", "1", "-framerate", str(fps), "-i", last_png,
                 "-i", clip_b,
                 "-filter_complex", filtergraph,
                 "-map", "[out]", "-map", "1:a?",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "12",
                 "-c:a", "copy", blended],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                log.warning("seam %d: blend failed (exit %s): %s; keeping the hard cut",
                            si, r.returncode, (r.stderr or "")[-400:])
                continue

            clip_paths[idx + 1] = blended
            done += 1
            log.info("seam %d: fill blended over %d frames at clips %d->%d (slot %dx%d+%d+%d)",
                     si, n_frames, idx, idx + 1, sw, sh, sx, sy)
        except Exception as e:
            log.warning("seam %d: %s; keeping the hard cut", si, e)
    return done
