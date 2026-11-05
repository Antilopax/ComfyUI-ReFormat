"""
The ReFormat nodes: Scene Split -> Scene Clip -> Canvas + Pad -> (your outpaint
graph) -> Save Chunk, repeated per chunk, then Stitch.

All five run on the CPU. They do, in ComfyUI, what the CPU worker and the
job handler do around the GPU graph of antilopax.ai's ReFormat tool.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.request

import torch

from . import media
from . import scenes as sc

log = logging.getLogger("ReFormat")

try:  # inside ComfyUI
    import folder_paths
except ImportError:  # standalone (tests)
    folder_paths = None

try:
    from comfy.utils import common_upscale as _comfy_upscale
except ImportError:
    _comfy_upscale = None

try:  # an output holding a blocker silently skips every node that reads it
    from comfy_execution.graph_utils import ExecutionBlocker  # light module, no torch/nodes import
except ImportError:
    try:  # older ComfyUI
        from comfy_execution.graph import ExecutionBlocker
    except ImportError:  # standalone (tests)
        class ExecutionBlocker:
            def __init__(self, message):
                self.message = message

CATEGORY = "ReFormat"
SCENES_T = "REFORMAT_SCENES"
SLOT_T = "REFORMAT_SLOT"
MANIFEST_VERSION = 1
SCALE_METHODS = ("bicubic", "lanczos", "bilinear", "area")


# ─── paths ─────────────────────────────────────────────────────────────────────

def _output_root() -> str:
    """Where work folders go: REFORMAT_WORK_DIR when set (a bigger disk, or tests), else ComfyUI's output."""
    env = os.environ.get("REFORMAT_WORK_DIR")
    if env:
        return os.path.abspath(env)
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.abspath("output")


def _input_root() -> str:
    env = os.environ.get("REFORMAT_INPUT_DIR")
    if env:
        return os.path.abspath(env)
    if folder_paths is not None:
        return folder_paths.get_input_directory()
    return os.path.abspath("input")


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._") or "video"


def _list_input_videos() -> list[str]:
    root = _input_root()
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(n for n in names
                  if n.lower().endswith(media.VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(root, n)))


def _resolve_video(video: str, video_path: str = "") -> str:
    p = (video_path or "").strip().strip('"')
    if p:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"video_path not found: {p}")
        return p
    if not video:
        raise FileNotFoundError("No video selected: put a video into ComfyUI's input folder or set video_path")
    if folder_paths is not None:
        p = folder_paths.get_annotated_filepath(video)
    else:
        p = video if os.path.isabs(video) else os.path.join(_input_root(), video)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"video not found: {p}")
    return p


def _rel_subfolder(path: str) -> str | None:
    """Subfolder for ComfyUI's /view previews; None when the folder is not under ComfyUI's output dir."""
    if folder_paths is None:
        return None
    try:
        rel = os.path.relpath(path, folder_paths.get_output_directory())
    except ValueError:  # Windows: different drives
        return None
    if rel.startswith("..") or os.path.isabs(rel):
        return None
    return rel.replace("\\", "/")


def _preview(files: list[str], folder: str) -> dict:
    """Thumbnails for the node, through ComfyUI's temp folder like its own Preview Image. Reported as outputs
    they would be a job's tile in the Assets panel, with the final video buried behind them."""
    sub = _rel_subfolder(folder)
    if sub is None:
        return {"images": []}
    temp = os.path.join(folder_paths.get_temp_directory(), sub)
    os.makedirs(temp, exist_ok=True)
    for f in files:
        shutil.copyfile(f, os.path.join(temp, os.path.basename(f)))
    return {"images": [{"filename": os.path.basename(f), "subfolder": sub, "type": "temp"} for f in files]}


def _video_preview(path: str) -> dict:
    """What ComfyUI's own Save Video returns: the frontend plays the file on the node and lists it in Assets."""
    sub = _rel_subfolder(os.path.dirname(path))
    if sub is None:
        return {"images": []}
    return {"images": [{"filename": os.path.basename(path), "subfolder": sub, "type": "output"}],
            "animated": (True,)}


def _file_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "missing"


def _scene_at(scenes: dict, index: int) -> dict:
    lst = scenes.get("scenes") or []
    if not lst:
        raise ValueError("The scenes input is empty")
    if index < 0 or index >= len(lst):
        raise ValueError(f"index {index} is out of range: this video has {len(lst)} chunks (0..{len(lst) - 1})")
    return lst[index]


def _chunk_path(scenes: dict, index: int) -> str:
    return os.path.join(scenes["work_dir"], "out", f"chunk_{index:03d}.mp4")


def _sidecar_path(scenes: dict, index: int) -> str:
    return os.path.join(scenes["work_dir"], "out", f"chunk_{index:03d}.json")


def _read_sidecar(scenes: dict, index: int) -> dict:
    try:
        with open(_sidecar_path(scenes, index), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _set_render_ids(manifest: dict) -> None:
    """Give every chunk an id of what its render depends on: the source, the delivery size and its own span.

    Save Chunk stores it in the sidecar. After a new split with other settings, a chunk
    whose span moved gets a new id, so its old render no longer counts as rendered;
    chunks that did not change keep theirs.
    """
    common = [os.path.basename(manifest["source_path"]), _file_size(manifest["source_path"]),
              manifest["direction"], manifest["clip_fps"],
              manifest["delivery"]["width"], manifest["delivery"]["height"]]
    for s in manifest["scenes"]:
        key = common + [round(s["start_time"], 3), round(s["end_time"], 3), s["frame_count"],
                        s["auto_split"], s["chunk"], s["chunk_count"]]
        s["render_id"] = hashlib.sha1(json.dumps(key).encode("utf-8")).hexdigest()[:16]


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def _render_settings(prompt: dict | None) -> str | None:
    """Fingerprint of the settings a render depends on, read from the queued graph: every node upstream of
    Save Chunk's images (prompt, seed, steps, LoRA, canvas), so it works for any graph and needs no wiring.

    Scene Clip's index moves on every run. Scene Split has its own check, the render ids, which keeps the
    chunks a new split did not touch.
    """
    if not prompt:
        return None
    todo = [n["inputs"]["images"][0] for n in prompt.values()
            if n.get("class_type") == "ReFormatSaveChunk" and isinstance(n.get("inputs", {}).get("images"), list)]
    seen = {}
    while todo:
        node_id = todo.pop()
        node = prompt.get(node_id)
        if node_id in seen or node is None or node.get("class_type") == "ReFormatSceneSplit":
            continue
        inputs = dict(node.get("inputs") or {})
        if node.get("class_type") == "ReFormatSceneClip":
            inputs.pop("index", None)
        seen[node_id] = [node.get("class_type"), inputs]
        todo += [v[0] for v in inputs.values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
    return hashlib.sha1(json.dumps(seen, sort_keys=True).encode("utf-8")).hexdigest()[:16] if seen else None


def _run_settings(prompt: dict | None) -> str | None:
    """The settings chunks must carry to count as rendered in this run. None (any render counts) without a
    graph, and when Scene Clip's index is a number: then one chunk is redone on purpose, the rest stay."""
    for node in (prompt or {}).values():
        if node.get("class_type") == "ReFormatSceneClip":
            index = (node.get("inputs") or {}).get("index")
            if isinstance(index, int) and index < 0:
                return _render_settings(prompt)
    return None


def _rendered(scenes: dict, index: int, settings: str | None = None) -> bool:
    """Chunk `index` has a render made for its current span and, when `settings` is given, with those settings.

    A leftover from an earlier split of the same video whose chunk at this
    index covered other frames does not count. Sidecars without a render id
    (written before it existed) are taken at face value.
    """
    if not os.path.isfile(_chunk_path(scenes, index)):
        return False
    side = _read_sidecar(scenes, index)
    if settings is not None and side.get("settings") != settings:
        return False
    want = _scene_at(scenes, index).get("render_id")
    got = side.get("render_id")
    return not want or not got or got == want


def _free_path(path: str) -> str:
    """`path`, or the next free final_001.mp4 next to it: a result that cost GPU time is never overwritten."""
    base, ext = os.path.splitext(path)
    n = 0
    while os.path.exists(path):
        n += 1
        path = f"{base}_{n:03d}{ext}"
    return path


_last_missing: dict[str, tuple] = {}  # work folder -> (settings, chunks missing after the last continue)


def _made_progress(work_dir: str, settings: str, missing: int) -> bool:
    """False when the last continue left as many chunks missing as before: a graph that never completes a
    chunk must not loop on a rented GPU."""
    last = _last_missing.get(work_dir)
    _last_missing[work_dir] = (settings, missing)
    return not (last and last[0] == settings and missing >= last[1])


def _as_queued(prompt: dict) -> dict:
    """The graph as it was queued. ComfyUI writes each node's IS_CHANGED result into it while it runs and reuses
    that value when it finds one, so a copy that keeps them runs from the cache instead of rendering."""
    return {node_id: {k: v for k, v in node.items() if k != "is_changed"} for node_id, node in prompt.items()}


def _queue_next_run(prompt: dict, extra_pnginfo: dict | None) -> bool:
    """Queue the same graph again through ComfyUI's own HTTP API, which needs no open browser tab and is the
    same on every ComfyUI version. Nothing to do when a batch count already queued it."""
    try:
        from server import PromptServer
        srv = PromptServer.instance
        graph = _as_queued(prompt)
        if any(task[2] == graph for task in srv.prompt_queue.get_current_queue_volatile()[1]):
            return True
        host = "127.0.0.1" if srv.address in ("0.0.0.0", "::", "") else srv.address
        body = {"prompt": graph, "client_id": srv.client_id}
        if extra_pnginfo:
            body["extra_data"] = {"extra_pnginfo": extra_pnginfo}
        req = urllib.request.Request(f"http://{host}:{srv.port}/prompt", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30):
            return True
    except Exception as e:
        log.warning("could not queue the next chunk (%s); press Run to go on", e)
        return False


def _same_shot(a: dict, b: dict) -> bool:
    """Two adjacent chunks cut from one continuous shot."""
    return bool(a.get("auto_split")) and bool(b.get("auto_split")) and a.get("group") == b.get("group")


def _upscale(images: torch.Tensor, w: int, h: int, method: str) -> torch.Tensor:
    if _comfy_upscale is not None:
        return _comfy_upscale(images.movedim(-1, 1), w, h, method, "disabled").movedim(1, -1)
    import torch.nn.functional as F
    mode = {"bilinear": "bilinear", "bicubic": "bicubic", "area": "area", "lanczos": "bicubic"}[method]
    kw = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(images.movedim(-1, 1), size=(h, w), mode=mode, **kw).movedim(1, -1).clamp(0, 1)


# ─── 1. Scene Split ────────────────────────────────────────────────────────────

class ReFormatSceneSplit:
    """Detect cuts, chunk long shots, normalise fps, cut clips, pull the audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_list_input_videos(), {"video_upload": True,
                                                 "tooltip": "A video from ComfyUI's input folder."}),
                "direction": (["auto", "landscape-to-portrait", "portrait-to-landscape", "custom"],
                              {"default": "auto",
                               "tooltip": "Which way the canvas grows. auto: a portrait video becomes landscape, "
                                          "anything else portrait, going by the size a player shows (rotation "
                                          "metadata of phone videos included). Pick by hand if a video is stored "
                                          "sideways without that metadata. Canvas + Pad follows this setting."}),
                "split_mode": (["detect", "single"],
                               {"default": "detect",
                                "tooltip": "detect: PySceneDetect finds the cuts. single: the whole video "
                                           "is one shot (skips detection, still chunks long videos)."}),
                "threshold": ("FLOAT", {"default": sc.DETECT_THRESHOLD, "min": 1.0, "max": 100.0, "step": 0.5,
                                        "tooltip": "ContentDetector threshold. Higher = fewer cuts."}),
                "min_scene_seconds": ("FLOAT", {"default": sc.MIN_SCENE_SECONDS, "min": 0.0, "max": 10.0, "step": 0.1,
                                                "tooltip": "Shorter detections are merged into a neighbour."}),
                "max_chunk_seconds": ("FLOAT", {"default": sc.MAX_CHUNK_SECONDS, "min": 1.0, "max": 60.0, "step": 0.5,
                                                "tooltip": "Longer shots are cut into equal chunks. 10 s at 30 fps "
                                                           "is the 1080p two-pass ceiling of a 32 GB GPU."}),
                "max_fps": ("FLOAT", {"default": sc.MAX_FPS, "min": 1.0, "max": 120.0, "step": 1.0,
                                      "tooltip": "Clips are halved (60 -> 30, 50 -> 25) until at or below this."}),
                "project": ("STRING", {"default": "",
                                       "tooltip": "Work folder under output/reformat/. Empty = the video's name."}),
            },
            "optional": {
                "video_path": ("STRING", {"default": "", "tooltip": "Absolute path; overrides the picker."}),
            },
        }

    RETURN_TYPES = (SCENES_T, "INT", "STRING", "STRING")
    RETURN_NAMES = ("scenes", "scene_count", "summary", "audio_path")
    FUNCTION = "split"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Cuts a video into scene clips the way antilopax.ai's ReFormat does: PySceneDetect cuts, "
                   "long shots chunked, fps halved to <= 30, audio kept aside for the stitch.")

    @classmethod
    def IS_CHANGED(cls, video="", video_path="", **kwargs):
        try:
            p = _resolve_video(video, video_path)
        except Exception:
            return float("nan")
        return f"{p}:{_file_sig(p)}"

    @classmethod
    def VALIDATE_INPUTS(cls, video="", video_path=""):
        # Only these two are taken over from ComfyUI's own checks (so video_path
        # may override the picker); every other input keeps the built-in range checks.
        try:
            _resolve_video(video, video_path)
        except Exception as e:
            return str(e)
        return True

    def split(self, video, direction, split_mode, threshold, min_scene_seconds, max_chunk_seconds, max_fps,
              project, video_path=""):
        media.require_ffmpeg()
        src = _resolve_video(video, video_path)
        name = _safe_name(project.strip() or os.path.splitext(os.path.basename(src))[0])
        work_dir = os.path.join(_output_root(), "reformat", name)
        os.makedirs(os.path.join(work_dir, "out"), exist_ok=True)

        info = media.get_video_info(src)
        log.info("source %dx%d @ %.3f fps, %.2fs, %s, audio=%s", info["width"], info["height"], info["fps"],
                 info["duration"], info["orientation"], info["has_audio"])
        picked = direction == "auto"
        if picked:
            direction = "portrait-to-landscape" if info["orientation"] == "portrait" else "landscape-to-portrait"

        delivery_path, dw, dh = media.scale_for_delivery(src, os.path.join(work_dir, "delivery.mp4"), info, direction)
        dinfo = media.get_video_info(delivery_path)
        log.info("delivery %dx%d%s", dw, dh, " (source passthrough)" if delivery_path == src else "")

        if split_mode == "single":
            detected = []
        else:
            with tempfile.TemporaryDirectory() as td:
                proxy = os.path.join(td, "proxy.mp4")
                media.write_detection_proxy(src, proxy, info, direction)
                detected = sc.detect_scenes(proxy, threshold=threshold, min_scene_length=min_scene_seconds)
        if not detected:
            detected = [{
                "index": 0, "start_time": 0.0, "end_time": dinfo["duration"],
                "start_timecode": "00:00:00.000", "end_timecode": str(dinfo["duration"]),
            }]

        clip_fps = sc.normalized_fps(dinfo["fps"], max_fps)
        if abs(clip_fps - dinfo["fps"]) > 0.01:
            log.info("normalising clips %.3f fps -> %.3f fps", dinfo["fps"], clip_fps)
        chunks = sc.subdivide_long_scenes(detected, max_chunk_seconds)

        audio_path = None
        if info["has_audio"]:
            try:
                audio_path = os.path.join(work_dir, "audio.aac")
                media.extract_audio(delivery_path, audio_path)
            except Exception as e:
                log.warning("audio extraction failed: %s", e)
                audio_path = None

        results = []
        thumbs = []
        for s in chunks:
            idx = s["index"]
            clip = os.path.join(work_dir, f"scene_{idx:03d}.mp4")
            media.split_scene(delivery_path, s["start_time"], s["end_time"], clip, fps=clip_fps)
            ci = media.get_video_info(clip)
            frames = media.count_frames(clip) or ci["frame_count"]
            thumb = os.path.join(work_dir, f"thumb_{idx:03d}.jpg")
            try:
                media.extract_thumbnail(delivery_path, s["start_time"] + (s["end_time"] - s["start_time"]) / 3,
                                        thumb, max_width=320)
                thumbs.append(thumb)
            except Exception as e:
                log.warning("thumbnail %d failed: %s", idx, e)
            results.append({
                "index": idx,
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "clip_path": clip,
                "thumbnail_path": thumb if os.path.exists(thumb) else None,
                "fps": ci["fps"],
                "frame_count": frames,
                "duration": ci["duration"],
                "width": ci["width"],
                "height": ci["height"],
                "auto_split": bool(s.get("auto_split")),
                "group": s.get("group", idx),
                "chunk": s.get("chunk", 0),
                "chunk_count": s.get("chunk_count", 1),
            })

        manifest = {
            "version": MANIFEST_VERSION,
            "work_dir": work_dir,
            "source_path": src,
            "direction": direction,
            "split_mode": split_mode,
            "clip_fps": clip_fps,
            "audio_path": audio_path,
            "source": {k: info[k] for k in ("width", "height", "fps", "duration", "has_audio", "orientation")},
            "delivery": {"path": delivery_path, "width": dw, "height": dh, "fps": dinfo["fps"],
                         "duration": dinfo["duration"]},
            "scenes": results,
        }
        _set_render_ids(manifest)
        with open(os.path.join(work_dir, "scenes.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        lines = [f"{direction}{' (auto)' if picked else ''}, source {info['width']}x{info['height']}"]
        lay = sc.canvas_layout(dw, dh, direction) if direction in sc.DIRECTIONS else None
        if lay and (lay["slot"]["w"], lay["slot"]["h"]) == (lay["canvas_width"], lay["canvas_height"]):
            lines.append("This video already fills that canvas, so there is nothing to outpaint. Try the other "
                         "direction.")
        lines.append(f"{len(results)} chunk(s) from {info['duration']:.1f}s, clips {dw}x{dh} @ {clip_fps:g} fps"
                     f"{', audio kept' if audio_path else ', no audio'}  ->  {work_dir}")
        for r in results:
            kind = (f"shot {r['group']} part {r['chunk'] + 1}/{r['chunk_count']}" if r["auto_split"]
                    else f"shot {r['group']}")
            lines.append(f"#{r['index']:02d}  {r['start_time']:7.2f}s - {r['end_time']:7.2f}s  "
                         f"{r['frame_count']:4d} frames  {kind}")
        summary = "\n".join(lines)
        log.info("%s", summary)

        return {"ui": _preview(thumbs, work_dir), "result": (manifest, len(results), summary, audio_path or "")}


# ─── 2. Scene Clip ─────────────────────────────────────────────────────────────

class ReFormatSceneClip:
    """One chunk as frames, plus what the continuation of a split shot needs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes": (SCENES_T,),
                # A string here is the control's default mode (the frontend's own PrimitiveInt does the same).
                "index": ("INT", {"default": -1, "min": -1, "max": 9999, "control_after_generate": "fixed",
                                  "tooltip": "-1 = auto: the first chunk that has no render with the current "
                                             "settings. Every run renders the next chunk, and Stitch queues the "
                                             "next run until the video is done. New prompt or seed = every chunk "
                                             "again. 0 or more = exactly this chunk, the others stay."}),
                "pad_to_8k1": ("BOOLEAN", {"default": True,
                                           "tooltip": "LTX generates 8k+1 frames; other lengths are silently "
                                                      "cut short. Repeats the last frame up to the next 8k+1. "
                                                      "Save Chunk trims the result back."}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "BOOLEAN", "BOOLEAN", "IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "frame_count", "true_frame_count", "fps", "continues_previous", "bypass_i2v",
                    "previous_last_frame", "index", "clip_path", "label")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Loads chunk `index` (-1 = the next chunk without a render). When the chunk continues a "
                   "shot whose previous chunk is already rendered, continues_previous is true and "
                   "previous_last_frame holds that render's last frame (the anchor for the img2video "
                   "conditioning). bypass_i2v is its inverse, ready for the LTX img2video node's bypass input; "
                   "index is the chunk actually loaded, for Save Chunk. In auto mode with every chunk rendered, "
                   "all outputs but `label` are blocked, so nothing downstream runs.")

    @classmethod
    def IS_CHANGED(cls, scenes=None, index=0, **kwargs):
        if not isinstance(scenes, dict):
            return float("nan")
        if index < 0:  # which chunk is next also depends on the settings, and IS_CHANGED does not get the graph
            return float("nan")
        try:
            s = _scene_at(scenes, index)
        except Exception:
            return float("nan")
        prev = _chunk_path(scenes, index - 1) if index > 0 else ""
        return f"{_file_sig(s['clip_path'])}|{_file_sig(prev) if prev else '-'}"

    def load(self, scenes, index, pad_to_8k1, prompt=None):
        media.require_ffmpeg()
        n_all = len(scenes.get("scenes") or [])
        auto = index < 0
        settings = _run_settings(prompt) if auto else None
        todo = []
        if auto:
            todo = [i for i in range(n_all) if not _rendered(scenes, i, settings)]
            if not todo:
                msg = (f"All {n_all} chunks are rendered with these settings, nothing to do. The final video is in "
                       f"the Assets panel and in {scenes['work_dir']}. To render the video again, change the prompt "
                       f"or seed; to render one chunk again, set index to its number.")
                log.info("%s", msg)
                return (ExecutionBlocker(None),) * 9 + (msg,)
            index = todo[0]
        scene = _scene_at(scenes, index)
        images, _ = media.decode_frames(scene["clip_path"])
        true_n = int(images.shape[0])
        if pad_to_8k1 and (true_n - 1) % 8 != 0:
            target = ((true_n - 1) // 8 + 1) * 8 + 1
            images = torch.cat([images, images[-1:].expand(target - true_n, -1, -1, -1)], dim=0)
            log.info("chunk %d: padded %d -> %d frames (8k+1)", index, true_n, target)

        continues = False
        prev_frame = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        if index > 0:
            prev = _scene_at(scenes, index - 1)
            if _same_shot(prev, scene):
                prev_out = _chunk_path(scenes, index - 1)
                if _rendered(scenes, index - 1, settings):
                    prev_frame = media.last_frame(prev_out)
                    continues = True
                else:
                    log.warning("chunk %d continues chunk %d, but chunk %d has not been rendered yet "
                                "(Save Chunk writes %s). Render it first for a seamless continuation.",
                                index, index - 1, index - 1, prev_out)

        label = (f"chunk {index + 1}/{n_all} (index {index}): {scene['start_time']:.2f}s - "
                 f"{scene['end_time']:.2f}s, {true_n} frames @ {scene['fps']:g} fps"
                 + (", continues the previous chunk" if continues else "")
                 + (f". {len(todo) - 1} more chunk(s) to render after this one" if auto else "")
                 + (". The settings changed, so older renders are replaced"
                    if any(_rendered(scenes, i) for i in todo) else ""))
        return (images, int(images.shape[0]), true_n, float(scene["fps"]), continues, not continues, prev_frame,
                int(index), scene["clip_path"], label)


# ─── 3. Canvas + Pad ───────────────────────────────────────────────────────────

class ReFormatCanvas:
    """Scale the band to the slot and pad it onto the preset canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "direction": (["auto"] + list(sc.DIRECTIONS),
                              {"default": "auto",
                               "tooltip": "auto follows Scene Split through the scenes input. Without that wire it "
                                          "goes by the frames: taller than wide becomes landscape."}),
                "split_percent": ("INT", {"default": 50, "min": 0, "max": 100,
                                          "tooltip": "Where the band sits: 0 = top/left edge, 50 = centred, "
                                                     "100 = bottom/right edge."}),
                "target_edge": ("INT", {"default": sc.TARGET_EDGE, "min": 512, "max": 4096, "step": 32,
                                        "tooltip": "Long edge of the canvas. 1920 gives 1088x1920 / 1920x1088."}),
                "scale_method": (list(SCALE_METHODS), {"default": "bicubic"}),
            },
            "optional": {
                "scenes": (SCENES_T,),
            },
        }

    RETURN_TYPES = ("IMAGE", SLOT_T, "INT", "INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "slot", "canvas_width", "canvas_height", "pad_top", "pad_bottom", "pad_left",
                    "pad_right", "layout")
    FUNCTION = "pad"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The hosted tool's canvas math: band scaled to the canvas's short axis (snapped to 32), "
                   "black padding where the model has to invent picture. Wire `slot` into Save Chunk so the "
                   "stitch knows which region was generated.")

    def pad(self, images, direction, split_percent, target_edge, scale_method, scenes=None):
        n, h, w, c = images.shape
        if direction == "auto":
            direction = (scenes or {}).get("direction")
            if direction not in sc.DIRECTIONS:
                direction = "portrait-to-landscape" if h > w else "landscape-to-portrait"
        lay = sc.canvas_layout(w, h, direction, split_percent, target_edge)
        slot = lay["slot"]
        if lay["clamped"]:
            log.warning("source %dx%d already fills the %s canvas; the band is squeezed to the slot %dx%d",
                        w, h, direction, slot["w"], slot["h"])
        if (w, h) != (slot["w"], slot["h"]):
            scaled = _upscale(images, slot["w"], slot["h"], scale_method)
        else:
            scaled = images
        canvas = torch.zeros((n, lay["canvas_height"], lay["canvas_width"], c), dtype=images.dtype)
        canvas[:, slot["y"]:slot["y"] + slot["h"], slot["x"]:slot["x"] + slot["w"], :] = scaled
        slot_out = {"x": slot["x"], "y": slot["y"], "w": slot["w"], "h": slot["h"],
                    "canvas_width": lay["canvas_width"], "canvas_height": lay["canvas_height"]}
        layout = (f"canvas {lay['canvas_width']}x{lay['canvas_height']}, band {w}x{h} -> slot "
                  f"{slot['w']}x{slot['h']} at ({slot['x']},{slot['y']}), pad T{lay['pad_top']} B{lay['pad_bottom']} "
                  f"L{lay['pad_left']} R{lay['pad_right']}")
        return (canvas, slot_out, lay["canvas_width"], lay["canvas_height"], lay["pad_top"], lay["pad_bottom"],
                lay["pad_left"], lay["pad_right"], layout)


# ─── 4. Save Chunk ─────────────────────────────────────────────────────────────

class ReFormatSaveChunk:
    """Write one rendered chunk where Scene Clip and Stitch expect it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scenes": (SCENES_T,),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999,
                                  "tooltip": "Same index as the Scene Clip this render came from."}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 35,
                                "tooltip": "h264 quality (lower = better). 19 is what the hosted graph saves with."}),
            },
            "optional": {
                "slot": (SLOT_T, {"tooltip": "From Canvas + Pad. Lets Stitch blend only the generated region."}),
                "true_frame_count": ("INT", {"default": 0, "min": 0, "max": 100000, "forceInput": True,
                                             "tooltip": "From Scene Clip; trims the 8k+1 padding back off."}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("chunk_path",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Saves the rendered chunk as output/reformat/<project>/out/chunk_NNN.mp4 (h264, no audio), "
                   "trimmed to the chunk's true length, with a sidecar holding the slot rect.")

    @classmethod
    def IS_CHANGED(cls, scenes=None, index=0, **kwargs):
        if not isinstance(scenes, dict):
            return float("nan")
        return _file_sig(_chunk_path(scenes, index))

    def save(self, images, scenes, index, crf, slot=None, true_frame_count=0, prompt=None):
        media.require_ffmpeg()
        scene = _scene_at(scenes, index)
        want = int(true_frame_count) if true_frame_count and true_frame_count > 0 else int(scene["frame_count"])
        n = int(images.shape[0])
        if n > want:
            images = images[:want]
            log.info("chunk %d: trimmed %d -> %d frames", index, n, want)
        elif n < want:
            log.warning("chunk %d: only %d of %d frames rendered; the stitched video will be short here",
                        index, n, want)

        out_dir = os.path.join(scenes["work_dir"], "out")
        os.makedirs(out_dir, exist_ok=True)
        path = _chunk_path(scenes, index)
        # a chunk redone on purpose stays part of the run it belongs to, whatever settings it was redone with
        settings = (_run_settings(prompt) or _read_sidecar(scenes, index).get("settings")
                    or _render_settings(prompt))
        media.encode_frames(images, path, fps=float(scene["fps"]), crf=crf)
        sidecar = {
            "index": index,
            "frames": int(images.shape[0]),
            "fps": float(scene["fps"]),
            "width": int(images.shape[2]),
            "height": int(images.shape[1]),
            "slot": dict(slot) if isinstance(slot, dict) else None,
            "render_id": scene.get("render_id"),
            "settings": settings,
            "written_at": time.time(),
        }
        with open(_sidecar_path(scenes, index), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        log.info("chunk %d saved: %s (%d frames @ %g fps)", index, path, sidecar["frames"], sidecar["fps"])

        previews = []
        thumb = os.path.join(out_dir, f"chunk_{index:03d}.jpg")
        try:
            media.write_png(images[0], thumb)  # ffmpeg picks the format from the extension
            previews.append(thumb)
        except Exception as e:
            log.warning("chunk %d: preview failed: %s", index, e)
        return {"ui": _preview(previews, out_dir), "result": (path,)}


# ─── 5. Stitch ─────────────────────────────────────────────────────────────────

class ReFormatStitch:
    """Concatenate the rendered chunks, soften auto-split seams, lay the original audio back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes": (SCENES_T,),
                "blend_seams": ("BOOLEAN", {"default": True,
                                            "tooltip": "Crossfade the generated region across the seam between "
                                                       "two chunks of one shot. The source band stays a hard "
                                                       "cut (it is continuous anyway)."}),
                "seam_frames": ("INT", {"default": sc.SEAM_FRAMES, "min": 2, "max": 30,
                                        "tooltip": "Length of the dissolve. 2 frames is enough when each "
                                                   "continuation chunk is anchored on the previous chunk's last "
                                                   "frame (the included workflow does that in both passes); "
                                                   "longer dissolves ghost moving fill."}),
                "filename": ("STRING", {"default": "final",
                                        "tooltip": "Written as output/reformat/<project>/<filename>.mp4"}),
            },
            "optional": {
                "chunk_paths": ("STRING", {"default": "", "multiline": True,
                                           "tooltip": "One rendered clip per line, in order, if you saved them "
                                                      "yourself instead of with Save Chunk."}),
                "slot": (SLOT_T, {"tooltip": "Fallback slot rect when a chunk has no sidecar."}),
                "saved_chunk": ("STRING", {"forceInput": True,
                                           "tooltip": "Wire Save Chunk's chunk_path here: Stitch then runs right "
                                                      "after every save and builds the final video as soon as "
                                                      "the last chunk is in. Until then it only reports progress."}),
                "render_all_chunks": ("BOOLEAN", {"default": True,
                                                  "tooltip": "One press of Run renders the whole video: with "
                                                             "saved_chunk wired and Scene Clip's index at -1, Stitch "
                                                             "queues the next run until every chunk is done. Cancel "
                                                             "the running job to stop, press Run to go on. Off = one "
                                                             "chunk per press of Run."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_path", "report")
    FUNCTION = "stitch"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Joins the rendered chunks and puts the original audio back. With Save Chunk's chunk_path "
                   "wired into saved_chunk it runs after every save, queues the next chunk and builds the final "
                   "video once every chunk is rendered; on its own it needs every chunk rendered and refuses "
                   "otherwise. The final video plays on this node; an earlier result is never overwritten.")

    @classmethod
    def IS_CHANGED(cls, scenes=None, chunk_paths="", **kwargs):
        if not isinstance(scenes, dict):
            return float("nan")
        paths = [l.strip().strip('"') for l in (chunk_paths or "").splitlines() if l.strip()] or \
                [_chunk_path(scenes, s["index"]) for s in scenes.get("scenes") or []]
        return "|".join(_file_sig(p) for p in paths) + "|" + _file_sig(scenes.get("audio_path") or "")

    def stitch(self, scenes, blend_seams, seam_frames, filename, chunk_paths="", slot=None, saved_chunk=None,
               render_all_chunks=True, prompt=None, extra_pnginfo=None):
        media.require_ffmpeg()
        lst = scenes.get("scenes") or []
        if not lst:
            raise ValueError("The scenes input is empty")

        settings = None
        given = [l.strip().strip('"') for l in (chunk_paths or "").splitlines() if l.strip()]
        if given:
            if len(given) != len(lst):
                raise ValueError(f"chunk_paths lists {len(given)} clips, but the video has {len(lst)} chunks")
            paths = given
            missing = [i for i, p in enumerate(paths) if not os.path.isfile(p)]
        else:
            paths = [_chunk_path(scenes, s["index"]) for s in lst]
            settings = _run_settings(prompt)
            missing = [i for i in range(len(lst)) if not _rendered(scenes, i, settings)]
        if missing:
            if saved_chunk is not None:  # wired after Save Chunk: the loop is still running
                queued = (render_all_chunks and settings is not None
                          and _made_progress(scenes["work_dir"], settings, len(missing))
                          and _queue_next_run(prompt, extra_pnginfo))
                report = (f"{len(lst) - len(missing)}/{len(lst)} chunks rendered, still to go: {missing}. "
                          f"{'The next one is queued' if queued else 'Press Run for the next one'}, "
                          f"the final video is built after the last one.")
                log.info("%s", report)
                return {"ui": {"images": []}, "result": ("", report)}
            raise FileNotFoundError(
                f"Chunk(s) {missing} are not rendered yet (a render left over from an earlier split of this "
                f"video does not count). Run Scene Clip -> your graph -> Save Chunk for each index, then "
                f"stitch. Expected under {os.path.join(scenes['work_dir'], 'out')}")

        _last_missing.pop(scenes["work_dir"], None)
        seams = []
        unblended = []
        if blend_seams:
            for i in range(len(lst) - 1):
                if not _same_shot(lst[i], lst[i + 1]):
                    continue
                rect = (_read_sidecar(scenes, i + 1).get("slot") or _read_sidecar(scenes, i).get("slot")
                        or (dict(slot) if isinstance(slot, dict) else None))
                if rect:
                    seams.append({"after_clip": i, "slot": rect, "frames": int(seam_frames)})
                else:
                    unblended.append(i)
            if unblended:
                log.warning("seams after chunks %s have no slot rect (wire Canvas + Pad's slot into Save Chunk "
                            "or into this node); they stay hard cuts", unblended)

        out_path = _free_path(os.path.join(scenes["work_dir"], _safe_name(filename or "final") + ".mp4"))
        blended = 0
        with tempfile.TemporaryDirectory() as td:
            clips = list(paths)
            if seams:
                blended = media.blend_seams(clips, seams, td)
            with_audio = media.stitch_clips(clips, scenes.get("audio_path"), out_path)

        info = media.get_video_info(out_path)
        report = (f"{out_path}: {len(paths)} chunk(s), {info['width']}x{info['height']} @ {info['fps']:g} fps, "
                  f"{info['duration']:.2f}s, {blended}/{len(seams)} seam(s) blended, "
                  f"audio {'restored' if with_audio else 'none'}")
        log.info("%s", report)
        return {"ui": _video_preview(out_path), "result": (out_path, report)}


NODE_CLASS_MAPPINGS = {
    "ReFormatSceneSplit": ReFormatSceneSplit,
    "ReFormatSceneClip": ReFormatSceneClip,
    "ReFormatCanvas": ReFormatCanvas,
    "ReFormatSaveChunk": ReFormatSaveChunk,
    "ReFormatStitch": ReFormatStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReFormatSceneSplit": "Scene Split (ReFormat)",
    "ReFormatSceneClip": "Scene Clip (ReFormat)",
    "ReFormatCanvas": "Canvas + Pad (ReFormat)",
    "ReFormatSaveChunk": "Save Chunk (ReFormat)",
    "ReFormatStitch": "Stitch (ReFormat)",
}
