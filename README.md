# ComfyUI-ReFormat

Turn a 16:9 video into 9:16 (or the other way round) with an outpainting model, in ComfyUI, one chunk at a time.

Five CPU nodes do everything around the GPU graph: cut detection, chunking of long shots, fps normalisation, canvas and padding math, saving the rendered chunks, blending the seams and stitching the result back together with the original audio. This is the pipeline behind the ReFormat tool on [antilopax.ai](https://antilopax.ai/tools/reformat), with the same defaults, so you can run it on your own ComfyUI and GPU.

The workflow is the LTX-2.3 IC-LoRA outpaint graph the hosted tool renders with. [Download it from the tutorial page](https://antilopax.ai/tutorials/reformat), which also has setup tutorials for your own GPU and for a RunPod template with everything preloaded.

## Run a whole video

In the workflow: pick the video in Scene Split and press Run once. The direction is set for you: a portrait video becomes landscape and the other way round, rotation metadata of phone videos included, and the `summary` says what was picked. If a video is stored sideways without that metadata, set `direction` in Scene Split by hand. There is no batch count to set: each job renders the next chunk and queues the one after it, also with the browser tab closed, and the last one writes `output/reformat/<name>/final.mp4` with the original audio. The video then plays on the Stitch node and is the `final` tile in the **Assets** panel (press A), where its ... menu has Download. To stop, cancel the running job. After a stop, a failed run or a pod restart, press Run again: it carries on with the first chunk that has no render.

The progress bar fills once per chunk, so a full bar does not mean the video is done: Stitch's report (the **Result** box in the workflow) says how many chunks are in and which are still to go. From one run on an H100, as a rough guide: about 10 minutes of GPU time for a 23 s video with 7 chunks, loading the models included. Frame count drives the time, and other cards will differ.

To render the video again, change the prompt, seed or canvas and press Run: all chunks render again and the new result is saved next to the old one as `final_001.mp4`. Keep the seed on `fixed`: on `randomize` every press of Run starts the whole video over.

## The loop

An outpaint model renders one chunk of up to 10 s at a time. One job in the queue is one chunk, and Stitch queues the next job until all chunks are in:

1. **Scene Split**: pick a video from the input folder. PySceneDetect finds the cuts, shots longer than `max_chunk_seconds` are cut into equal chunks, the frame rate is halved until it is at or below 30 fps, and each chunk becomes a clip in `output/reformat/<name>/`. The audio is set aside for the stitch. The node shows one thumbnail per chunk, and `summary` lists the chunks. It runs once per video; later runs reuse its result.
2. **Scene Clip**: loads one chunk's frames (padded to 8k+1 frames for LTX). With `index` at **-1** (the default) it takes the first chunk that has no render with the current settings, so every run moves on by one chunk; with every chunk rendered it blocks everything downstream and `label` says so. The settings are everything upstream of Save Chunk (prompt, seed, canvas, sampler), so changing one of them makes all chunks due again. Set `index` to a number to render exactly that chunk again: it replaces that chunk, builds a new final video if the others are in, and queues nothing else. When the chunk continues a split shot and the previous chunk is rendered, `continues_previous` is true and `previous_last_frame` holds the last frame of that render: wire both into the img2video conditioning so the generated fill carries on instead of re-rolling at the seam. In a two-pass graph, anchor it in both passes: a latent upscale drops the first pass's anchor.
3. **Canvas + Pad**: scales the band to the canvas's short axis (snapped to 32) and pads it onto a 1088x1920 or 1920x1088 canvas. Outputs the padded frames, the slot rect and the pad values.
4. Your outpaint graph renders the canvas.
5. **Save Chunk**: writes `out/chunk_NNN.mp4`, trimmed back to the chunk's true length, plus a sidecar with the slot rect and the chunk's render id (source, size and span).
6. **Stitch**: concatenates the chunks, dissolves the generated region across the seam between two chunks of one shot (the source band stays a hard cut, it is continuous anyway), and lays the original audio back. Wire Save Chunk's `chunk_path` into `saved_chunk`: Stitch then runs after every save, reports how many chunks are still to go, queues the next job and builds the final video as soon as the last one is in. Without that wire it stitches on demand and refuses while a chunk is missing. The final video plays on the node. An earlier result is never overwritten: the next one is `final_001.mp4`, then `final_002.mp4`.

Changing the settings replaces the chunk renders in `out/`, only the final videos stay. To keep the chunks of an earlier version too, give Scene Split a new `project` name: a fresh work folder.

Scene Split and Stitch only ever need the CPU, so the same workflow runs on a laptop with the render steps muted when you only want the cut list.

## Install

- Clone this repository into `ComfyUI/custom_nodes/` and run `pip install -r requirements.txt` with ComfyUI's Python.
- `ffmpeg` and `ffprobe` must be on the PATH (ComfyUI-VideoHelperSuite needs the same).
- The workflow needs ComfyUI-LTXVideo and ComfyUI-VideoHelperSuite; the models are listed in the note inside the workflow.

## Parameters and defaults

| Node | Parameter | Default | What it does |
|---|---|---|---|
| Scene Split | `direction` | auto | auto: a portrait video becomes landscape, anything else portrait. Canvas + Pad follows it through its `scenes` input, so this is the only place to set it. |
| Scene Split | `threshold` | 38 | PySceneDetect ContentDetector threshold. Higher finds fewer cuts. |
| Scene Split | `min_scene_seconds` | 1.5 | Shorter detections are merged into a neighbour (motion blur, flashes). |
| Scene Split | `max_chunk_seconds` | 10 | Longer shots are cut into equal chunks. 10 s at 30 fps is the 1080p two-pass ceiling of a 32 GB GPU. |
| Scene Split | `max_fps` | 30 | Clips are halved (60 to 30, 50 to 25) until at or below this. Frame count, not seconds, drives VRAM and time. |
| Scene Split | `split_mode` | detect | `single` skips detection for a video you know is one shot; long videos are still chunked. |
| Canvas + Pad | `split_percent` | 50 | Where the band sits in the padding: 0 is the top or left edge, 100 the bottom or right edge. |
| Canvas + Pad | `target_edge` | 1920 | Long edge of the canvas. |
| Scene Clip | `index` | -1 | -1 = the first chunk without a render with the current settings; a number = that chunk, once. |
| Scene Clip | `pad_to_8k1` | on | LTX renders 8k+1 frames and silently drops the rest; the last frame is repeated up to the next 8k+1 and Save Chunk trims it off again. |
| Save Chunk | `crf` | 19 | h264 quality of the saved chunk. |
| Stitch | `seam_frames` | 2 | Length of the dissolve across a split-shot seam. 2 frames is enough when each continuation chunk is anchored on the previous chunk's last frame; a longer dissolve from that still frame ghosts moving fill. |
| Stitch | `render_all_chunks` | on | After each chunk, queue the next job until the video is done. Off: one chunk per press of Run. Needs `saved_chunk` wired and `index` at -1. |

Cut detection runs on a fixed 720 px proxy, so a given video always splits at the same points regardless of its resolution.

## Good to know

- **In a container (RunPod and the like), start ComfyUI with `--cache-classic`.** Its default cache keeps the results of every earlier job and only clears them when RAM runs low, but it reads the host's RAM, not the container's limit. On a rented GPU it never clears: memory grows chunk by chunk until the container is killed (exit code 137). The classic cache drops the previous job's results first, and the models stay loaded. Press Run after a restart to carry on.
- **Objects that enter later**: a chunk only sees its own frames, so something that enters the source frame later in a shot (a bag, a second person) is unknown to the fill until then and pops in when the source shows it. Name it in the prompt and the fill can have it from the start. In a same-seed test the object held from the earlier chunk on instead of popping in. The prompt goes to every chunk, so in a video with several shots it can turn up in the other shots too.
- **RAM**: ComfyUI holds frames as float32. A padded 1088x1920 frame is 25 MB; a 10 s chunk at 30 fps is about 7.5 GB per IMAGE tensor stage. Lower `max_chunk_seconds` if you run out of RAM.
- **Work folder**: everything lives in `output/reformat/<name>/` (`scenes.json`, the clips, `audio.aac`, `out/` with the rendered chunks). Set the environment variable `REFORMAT_WORK_DIR` to put work folders somewhere else; previews in the node then stay empty because ComfyUI only serves its own output folder.
- **Finding the result later**: the video on the Stitch node is gone after a page reload, the tile in Assets stays until ComfyUI restarts, and the file stays in the work folder.
- **Re-running Scene Split** with other settings overwrites the clips of the same name. Rendered chunks in `out/` are not deleted. A chunk whose span stayed the same keeps its render; one that now covers other frames (say, after a lower `max_chunk_seconds`) no longer counts: Scene Clip renders it again and Stitch waits for it.
- **Stitch settings alone** (`seam_frames`, `filename`) start no render, so with every chunk rendered Run does nothing. To build the video again with them, set `index` to a chunk's number: that renders this one chunk again and stitches.
- **Rendering out of order** works, except that a continuation chunk rendered before its predecessor cannot anchor to it; `continues_previous` is then false and the node says so in the log. Re-rendering one part of a split shot also changes the frame the next part was anchored on, so render the following parts of that shot again too.
- **Standalone test**: `python tests/test_pipeline.py` synthesises a 20 s clip and runs the whole loop with an identity "render", explicit and auto index (needs only ffmpeg, torch and scenedetect, no ComfyUI).

## How this relates to antilopax.ai

The hosted tool runs the same split and stitch on CPU workers and the same graph on rented GPUs, with the chain, retries, refunds and limits handled for you. If you would rather not set up the models, upload the video there instead.

## License

AGPL-3.0, see `LICENSE`. Copyright (C) 2026 Antilopax.

Use it for your own videos, commercial ones included: the licence covers this code, not what you render with it. If you change it and offer it as a hosted service, you have to publish your changes under the same licence. The models have their own licences, so check those too.

Cut detection by [PySceneDetect](https://www.scenedetect.com/), media handling by [ffmpeg](https://ffmpeg.org/).
