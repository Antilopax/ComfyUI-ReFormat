"""
Standalone pipeline test: no ComfyUI needed, only ffmpeg on PATH plus the
Python deps (torch, numpy, scenedetect[opencv]).

    python tests/test_pipeline.py <video.mp4> [work_dir]

Runs Scene Split -> Scene Clip -> Canvas + Pad -> Save Chunk (the "outpaint"
is an identity: the padded canvas is saved as is) -> Stitch, and checks the
counts, sizes and seams; then the auto index (-1) and the Stitch wired after
Save Chunk, the way the included workflow loops. The expectations below are for the synthetic clip
made by make_test_video(): 20 s, 60 fps, 1920x1080, a 440 Hz tone, hard cuts
at 3 s, 4 s and 16 s. At the production threshold (38) the detector reports
the cuts at 4 s and 16 s on this synthetic content, so the 12 s middle shot
is chunked into 4-10 s and 10-16 s: four chunks of 120/180/180/120 frames.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def make_test_video(path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=60:duration=3",
         "-f", "lavfi", "-i", "color=c=red:size=1920x1080:rate=60:duration=1",
         "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=60:duration=12",
         "-f", "lavfi", "-i", "smptebars=size=1920x1080:rate=60:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=20",
         "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
         "-map", "[v]", "-map", "4:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", path],
        check=True,
    )


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else ""
    work = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="reformat_test_")
    os.environ["REFORMAT_WORK_DIR"] = work
    os.makedirs(work, exist_ok=True)
    if not video:
        video = os.path.join(work, "source_landscape_60fps.mp4")
        make_test_video(video)

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch
    from reformat import media
    from reformat.nodes import (ReFormatCanvas, ReFormatSaveChunk, ReFormatSceneClip, ReFormatSceneSplit,
                                ReFormatStitch)
    from reformat.scenes import canvas_layout

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            failures.append(msg)

    # ── canvas math (pure) ────────────────────────────────────────────────────
    lay = canvas_layout(1088, 612, "landscape-to-portrait")
    check(lay["canvas_width"] == 1088 and lay["canvas_height"] == 1920, "l2p canvas 1088x1920")
    check(lay["slot"] == {"x": 0, "y": 656, "w": 1088, "h": 608}, f"l2p slot {lay['slot']}")
    lay = canvas_layout(1080, 1920, "portrait-to-landscape")
    check(lay["canvas_width"] == 1920 and lay["canvas_height"] == 1088, "p2l canvas 1920x1088")
    check(lay["slot"] == {"x": 656, "y": 0, "w": 608, "h": 1088}, f"p2l slot {lay['slot']}")
    lay = canvas_layout(1088, 612, "landscape-to-portrait", split_percent=0)
    check(lay["pad_top"] == 0 and lay["pad_bottom"] == 1312, "split 0 puts the band at the top")

    # ── 1. split ──────────────────────────────────────────────────────────────
    t0 = time.time()
    out = ReFormatSceneSplit().split(video="", direction="landscape-to-portrait", split_mode="detect",
                                     threshold=38.0, min_scene_seconds=1.5, max_chunk_seconds=10.0,
                                     max_fps=30.0, project="test", video_path=video)
    scenes, count, summary, audio_path = out["result"]
    print(summary)
    print(f"  split took {time.time() - t0:.1f}s")
    lst = scenes["scenes"]
    check(count == 4, f"4 chunks (got {count})")
    if count == 4:
        check([s["frame_count"] for s in lst] == [120, 180, 180, 120],
              f"frame counts 120/180/180/120 (got {[s['frame_count'] for s in lst]})")
        check([round(s["start_time"], 2) for s in lst] == [0.0, 4.0, 10.0, 16.0],
              f"chunk starts 0/4/10/16 s (got {[round(s['start_time'], 2) for s in lst]})")
        check([s["auto_split"] for s in lst] == [False, True, True, False], "auto_split flags")
        check([s["group"] for s in lst] == [0, 1, 1, 2], f"shot groups (got {[s['group'] for s in lst]})")
        check(all(abs(s["fps"] - 30.0) < 0.01 for s in lst), "clips at 30 fps")
        check(all(s["width"] == 1088 and s["height"] == 612 for s in lst), "clips 1088x612")
    check(bool(audio_path) and os.path.getsize(audio_path) > 1000, "audio extracted")
    check(os.path.isfile(os.path.join(scenes["work_dir"], "scenes.json")), "manifest written")
    ids = [s.get("render_id") for s in lst]
    check(all(len(i or "") == 16 for i in ids) and len(set(ids)) == len(ids), "every chunk has its own render id")
    check(scenes["work_dir"].startswith(os.path.abspath(work)), f"work dir honours REFORMAT_WORK_DIR ({scenes['work_dir']})")
    check(all(os.path.isfile(s["thumbnail_path"]) for s in lst), "one thumbnail per chunk")

    # ── 2..4. per chunk: clip -> canvas -> save (identity outpaint) ──────────
    clip_node, canvas_node, save_node = ReFormatSceneClip(), ReFormatCanvas(), ReFormatSaveChunk()
    for i in range(count):
        images, n, true_n, fps, continues, bypass, prev_frame, idx, clip_path, label = clip_node.load(scenes, i, True)
        print("  " + label)
        check(bypass == (not continues) and idx == i, f"chunk {i}: bypass_i2v and index echo")
        check(tuple(images.shape[1:]) == (612, 1088, 3), f"chunk {i}: frames are 612x1088x3")
        check(true_n == lst[i]["frame_count"], f"chunk {i}: true frames {true_n}")
        check((n - 1) % 8 == 0 and n >= true_n, f"chunk {i}: padded to 8k+1 ({n})")
        expect_cont = i == 2  # chunk 2 continues chunk 1 (same shot); chunk 3 is a new shot
        check(continues == expect_cont, f"chunk {i}: continues_previous={continues}")
        if continues:
            check(tuple(prev_frame.shape) == (1, 1920, 1088, 3), "anchor is the previous render's last frame")

        canvas, slot, cw, ch, pt, pb, pl, pr, layout = canvas_node.pad(images, "landscape-to-portrait", 50, 1920,
                                                                       "bicubic")
        if i == 0:
            print("  " + layout)
        check(tuple(canvas.shape[1:]) == (1920, 1088, 3), f"chunk {i}: canvas 1088x1920")
        check((pt, pb, pl, pr) == (656, 656, 0, 0), f"chunk {i}: pads {(pt, pb, pl, pr)}")
        band = canvas[0, slot["y"]:slot["y"] + slot["h"], slot["x"]:slot["x"] + slot["w"], :]
        check(float(band.mean()) > 0.05 and float(canvas[0, :slot["y"], :, :].abs().max()) == 0.0,
              f"chunk {i}: band placed, padding black")

        res = save_node.save(canvas, scenes, i, 19, slot=slot, true_frame_count=true_n)
        path = res["result"][0]
        check(media.count_frames(path) == true_n, f"chunk {i}: saved {media.count_frames(path)} frames")

    # ── 5. stitch ─────────────────────────────────────────────────────────────
    seam_default = ReFormatStitch.INPUT_TYPES()["required"]["seam_frames"][1]["default"]
    check(seam_default == 2, f"seam_frames defaults to 2 (got {seam_default})")
    res = ReFormatStitch().stitch(scenes, True, seam_default, "final")
    final_path, report = res["result"]
    print("  " + report)
    info = media.get_video_info(final_path)
    check(media.count_frames(final_path) == 600, f"final has 600 frames (got {media.count_frames(final_path)})")
    check(info["width"] == 1088 and info["height"] == 1920, "final is 1088x1920")
    check(info["has_audio"], "final has audio")
    check(abs(info["duration"] - 20.0) < 0.15, f"final duration {info['duration']:.2f}s")
    check("1/1 seam(s) blended" in report, "exactly one seam blended (chunks 1->2)")
    from reformat import nodes as nodes_mod
    keep_rel = nodes_mod._rel_subfolder
    nodes_mod._rel_subfolder = lambda path: "reformat/x"  # as if the work folder sat in ComfyUI's output
    shown = nodes_mod._video_preview(final_path)
    nodes_mod._rel_subfolder = keep_rel
    check(shown == {"images": [{"filename": "final.mp4", "subfolder": "reformat/x", "type": "output"}],
                    "animated": (True,)}, f"Stitch reports the final as a video output (got {shown})")
    check(res["ui"] == {"images": []}, "a work folder outside ComfyUI's output has no video preview")
    check(not os.path.exists(os.path.join(os.path.dirname(final_path), "final.jpg")), "no leftover still next to it")

    # ── stitch refuses missing chunks ─────────────────────────────────────────
    os.replace(os.path.join(scenes["work_dir"], "out", "chunk_003.mp4"),
               os.path.join(scenes["work_dir"], "out", "chunk_003.mp4.bak"))
    try:
        ReFormatStitch().stitch(scenes, True, 16, "final2")
        check(False, "stitch must refuse a missing chunk")
    except FileNotFoundError as e:
        check("[3]" in str(e), f"stitch names the missing chunk: {e}")
    finally:
        os.replace(os.path.join(scenes["work_dir"], "out", "chunk_003.mp4.bak"),
                   os.path.join(scenes["work_dir"], "out", "chunk_003.mp4"))

    # ── auto index (-1): the next chunk without a render, stitched after the last ─
    from reformat.nodes import ExecutionBlocker
    out_dir = os.path.join(scenes["work_dir"], "out")
    clip_default = ReFormatSceneClip.INPUT_TYPES()["required"]["index"][1]
    check(clip_default["default"] == -1 and clip_default["control_after_generate"] == "fixed",
          "Scene Clip index defaults to -1 (auto) with a fixed control")
    with open(os.path.join(out_dir, "chunk_000.json"), encoding="utf-8") as f:
        check(json.load(f).get("render_id") == lst[0]["render_id"], "Save Chunk records the chunk's render id")
    res = clip_node.load(scenes, -1, True)
    check(all(isinstance(v, ExecutionBlocker) for v in res[:9]) and res[9].startswith("All 4 chunks"),
          f"auto with every chunk rendered blocks the render, label says why: {res[9]}")

    moved = []
    for i in (2, 3):  # take chunks 2 and 3 away: auto renders 2 (a continuation), then 3, then stops
        for ext in (".mp4", ".json"):
            p = os.path.join(out_dir, f"chunk_{i:03d}{ext}")
            os.replace(p, p + ".bak")
            moved.append(p + ".bak")
    stitch_node = ReFormatStitch()
    for expect in (2, 3):
        images, n, true_n, fps, continues, bypass, prev_frame, idx, clip_path, label = clip_node.load(scenes, -1, True)
        print("  " + label)
        check(idx == expect, f"auto picks chunk {expect} (got {idx})")
        check(continues == (expect == 2), f"auto chunk {expect}: continues_previous={continues}")
        canvas, slot = canvas_node.pad(images, "landscape-to-portrait", 50, 1920, "bicubic")[:2]
        saved = save_node.save(canvas, scenes, idx, 19, slot=slot, true_frame_count=true_n)["result"][0]
        final_path, report = stitch_node.stitch(scenes, True, seam_default, "final_auto",
                                                saved_chunk=saved)["result"]
        print("  " + report)
        if expect == 2:
            check(final_path == "" and "still to go: [3]" in report, "wired Stitch waits for the last chunk")
        else:
            check(os.path.isfile(final_path) and media.count_frames(final_path) == 600,
                  "wired Stitch builds the final video after the last chunk")
    res = clip_node.load(scenes, -1, True)
    check(isinstance(res[0], ExecutionBlocker), "a run after the last chunk does nothing")

    # ── new settings render again, without a new project ──────────────────────
    from reformat import nodes as nd

    def graph(seed=1, index=-1, preview="a"):
        return {
            "1": {"class_type": "ReFormatSceneSplit", "inputs": {"project": "test"}},
            "2": {"class_type": "ReFormatSceneClip", "inputs": {"scenes": ["1", 0], "index": index, "pad_to_8k1": True}},
            "3": {"class_type": "ReFormatCanvas", "inputs": {"images": ["2", 0], "split_percent": 50}},
            "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "7": {"class_type": "Sampler", "inputs": {"images": ["3", 0], "noise": ["6", 0]}},
            "4": {"class_type": "ReFormatSaveChunk", "inputs": {"images": ["7", 0], "index": ["2", 7], "crf": 19}},
            "5": {"class_type": "ReFormatStitch", "inputs": {"saved_chunk": ["4", 0]}},
            "8": {"class_type": "PreviewAny", "inputs": {"source": ["5", 1], "note": preview}},
        }

    sig = nd._render_settings(graph())
    check(sig and sig == nd._render_settings(graph(index=3)), "the chunk index is not part of the settings")
    check(sig == nd._render_settings(graph(preview="b")), "a node after the render is not part of the settings")
    check(sig != nd._render_settings(graph(seed=2)), "a new seed is a new render")
    check(nd._run_settings(graph(index=3)) is None, "a fixed index renders on purpose, whatever is there")
    check(nd._render_settings(None) is None, "no graph, no settings check")
    ran = graph()
    ran["2"]["is_changed"] = [float("nan")]
    check(nd._as_queued(ran) == graph(), "the next run is queued without ComfyUI's cached IS_CHANGED values")
    check(nd._render_settings(ran) == sig, "those values are not part of the settings either")

    def render_one(g, expect, name="final_new"):
        images, _, true_n, _, _, _, _, idx, _, label = clip_node.load(scenes, g["2"]["inputs"]["index"], True, prompt=g)
        check(idx == expect, f"picks chunk {expect} (got {idx}): {label}")
        canvas, slot = canvas_node.pad(images, "landscape-to-portrait", 50, 1920, "bicubic")[:2]
        saved = save_node.save(canvas, scenes, idx, 19, slot=slot, true_frame_count=true_n, prompt=g)["result"][0]
        return stitch_node.stitch(scenes, True, seam_default, name, saved_chunk=saved, prompt=g)["result"], label

    queued = []
    nd._queue_next_run = lambda prompt, extra: queued.append(prompt) or True
    p1, p2 = graph(seed=1), graph(seed=2)
    for i in range(4):
        (first, report), _ = render_one(p1, i)
    check(len(queued) == 3, f"one press of Run: Stitch queues the next run after chunks 0-2 (got {len(queued)})")
    check("The next one is queued" not in report and os.path.basename(first) == "final_new.mp4",
          f"the last chunk builds the final: {report}")
    check(isinstance(clip_node.load(scenes, -1, True, prompt=p1)[0], ExecutionBlocker), "same settings: nothing to do")
    queued.clear()
    (done, report), label = render_one(p2, 0)
    check("settings changed" in label and done == "" and "still to go: [1, 2, 3]" in report and len(queued) == 1,
          f"a new seed starts over at chunk 0 and carries on by itself: {label} / {report}")
    for i in (1, 2, 3):
        (second, _), _ = render_one(p2, i)
    check(os.path.basename(second) == "final_new_001.mp4" and os.path.isfile(first),
          f"the new result does not overwrite the old one (got {second})")
    check(isinstance(clip_node.load(scenes, -1, True, prompt=p2)[0], ExecutionBlocker), "p2 complete: nothing to do")

    queued.clear()
    redo = graph(seed=3, index=2)
    (third, report), _ = render_one(redo, 2)
    check(os.path.basename(third) == "final_new_002.mp4" and not queued,
          f"a fixed index redoes that chunk, builds right away and queues nothing ({report})")
    check(isinstance(clip_node.load(scenes, -1, True, prompt=p2)[0], ExecutionBlocker),
          "a chunk redone on purpose stays part of its run")

    # never a loop: a run that leaves as many chunks missing as the one before stops continuing
    os.replace(_chunk := os.path.join(out_dir, "chunk_003.mp4"), _chunk + ".hold")
    queued.clear()
    for _ in range(2):
        stitch_node.stitch(scenes, True, seam_default, "x", saved_chunk="y", prompt=p2)
    check(len(queued) == 1, f"no progress, no further run (queued {len(queued)})")
    queued.clear()
    report = stitch_node.stitch(scenes, True, seam_default, "x", saved_chunk="y", prompt=graph(seed=9),
                                render_all_chunks=False)["result"][1]
    check(not queued and "Press Run" in report, f"render_all_chunks off: one chunk per Run ({report})")
    os.replace(_chunk + ".hold", _chunk)

    # a render left over from another split of the video (chunk 1 covered other frames) does not count
    side = os.path.join(out_dir, "chunk_001.json")
    with open(side, encoding="utf-8") as f:
        keep = f.read()
    with open(side, "w", encoding="utf-8") as f:
        json.dump(dict(json.loads(keep), render_id="another-split"), f)
    idx = clip_node.load(scenes, -1, True)[7]
    check(idx == 1, f"auto re-renders a chunk saved for another split (got {idx})")
    cont = clip_node.load(scenes, 2, True)[4]
    check(cont is False, "no anchor from a chunk saved for another split")
    with open(side, "w", encoding="utf-8") as f:
        f.write(keep)
    for p in moved:
        os.remove(p)

    # a new split with shorter chunks: chunk 0 (0-4 s, its own shot) keeps its render, the re-chunked shot does not
    resplit = ReFormatSceneSplit().split(video="", direction="landscape-to-portrait", split_mode="detect",
                                         threshold=38.0, min_scene_seconds=1.5, max_chunk_seconds=5.0,
                                         max_fps=30.0, project="test", video_path=video)["result"][0]
    starts = [round(s["start_time"], 2) for s in resplit["scenes"]]
    check(starts == [0.0, 4.0, 8.0, 12.0, 16.0], f"re-split at 5 s chunks starts 0/4/8/12/16 (got {starts})")
    idx = clip_node.load(resplit, -1, True)[7]
    check(idx == 1, f"after the re-split auto keeps chunk 0 and starts at chunk 1 (got {idx})")

    # ── direction: auto, one field ───────────────────────────────────────────
    def split_auto(path, project):
        return ReFormatSceneSplit().split(video="", direction="auto", split_mode="single", threshold=38.0,
                                          min_scene_seconds=1.5, max_chunk_seconds=10.0, max_fps=30.0,
                                          project=project, video_path=path)["result"]

    def ffmpeg(*args):
        subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)

    short = os.path.join(work, "short_landscape.mp4")
    ffmpeg("-i", video, "-t", "2", "-vf", "scale=1280:720", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", short)
    wide = split_auto(short, "auto_wide")
    check(wide[0]["direction"] == "landscape-to-portrait", f"auto: landscape video goes portrait ({wide[0]['direction']})")
    check(wide[2].startswith("landscape-to-portrait (auto), source 1280x720"), f"summary names the pick ({wide[2][:60]})")
    band = torch.zeros((1, 612, 1088, 3))
    check(canvas_node.pad(band, "auto", 50, 1920, "bicubic", scenes=wide[0])[2:4] == (1088, 1920), "Canvas auto follows Scene Split")

    # a phone clip: stored wide, shown tall through rotation metadata; the file is copied, only the metadata changes
    tall = os.path.join(work, "phone_rotated.mp4")
    try:
        ffmpeg("-display_rotation", "90", "-i", short, "-c", "copy", tall)
    except subprocess.CalledProcessError:  # ffmpeg before 6 has no -display_rotation
        ffmpeg("-i", short, "-c", "copy", "-metadata:s:v:0", "rotate=90", tall)
    phone = split_auto(tall, "auto_phone")
    check(phone[0]["direction"] == "portrait-to-landscape", f"auto: rotated phone clip goes landscape ({phone[0]['direction']})")
    frames = clip_node.load(phone[0], 0, False)[0]
    check(frames.shape[1] > frames.shape[2], f"rotated phone clip decodes upright, taller than wide ({tuple(frames.shape)})")
    p_lay = canvas_node.pad(frames, "auto", 50, 1920, "bicubic", scenes=phone[0])
    check(p_lay[2:4] == (1920, 1088), f"Canvas follows it to 1920x1088 ({p_lay[2:4]})")

    tall_band = torch.zeros((1, 640, 360, 3))
    check(canvas_node.pad(tall_band, "auto", 50, 1920, "bicubic")[2:4] == (1920, 1088), "Canvas auto without the wire: by the frames")
    check(canvas_node.pad(tall_band, "landscape-to-portrait", 50, 1920, "bicubic", scenes=phone[0])[2:4] == (1088, 1920),
          "a direction set on Canvas stays in charge")

    wrong = ReFormatSceneSplit().split(video="", direction="landscape-to-portrait", split_mode="single", threshold=38.0,
                                       min_scene_seconds=1.5, max_chunk_seconds=10.0, max_fps=30.0,
                                       project="auto_wrong", video_path=tall)["result"]
    check("nothing to outpaint" in wrong[2], "the wrong way round says there is nothing to outpaint")

    print()
    print(f"work dir: {work}")
    if failures:
        print(f"{len(failures)} FAILED: " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
