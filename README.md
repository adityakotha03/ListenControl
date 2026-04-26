# ListenControl -- Listener Facial Motion Generation

Predict and render listener facial reactions (FLAME parameters) from speaker audio, optionally controlled by free-form text prompts.

## Available Models

| Model | Alias(es) | Default Weights | Description |
|---|---|---|---|
| `bidir_cross_transformer` | `v5`, `transformer`, `bidir` | `best_bidir_cross_transformer.pt` | BidirCrossTransformer (Pipeline 5) -- best base reconstruction |
| `emotion_conditioned_transformer` | `v6`, `v6.1`, `p6`, `emo_transformer` | `best_emo_transformer_v2.pt` (falls back to `best_emo_transformer.pt`) | CLIP text-conditioned via FiLM + CFG (Pipeline 6.1) |
| `128` | -- | `best_model_dim_128_30.pt` | Legacy LSTM (small) |
| `256` | -- | `best_model_dim_256_30.pt` | Legacy LSTM (large) |

## Directory Layout

```
runpod/
  main.py              # Inference CLI + pipeline entry point
  handler.py           # RunPod serverless handler
  model/
    architecture.py    # All model classes
  render/
    render_pipeline.py # FLAME mesh rendering (PyTorch3D)
    flame/             # FLAME assets (generic_model.pkl, etc.)
  weights/             # Place .pt checkpoint files here
  samples/             # Example .npz + .wav pairs
  outputs/             # Rendered videos go here
```

## Setup (RunPod / Local)

```bash
cd runpod
pip install -r requirements.txt
```

The runtime must use PyTorch `>=2.6`. Newer `transformers` releases block `torch.load` on older PyTorch builds because of CVE-2025-32434; rebuild the Docker image if you see that error.

PyTorch3D is needed for rendering. On RunPod with CUDA:
```bash
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt221/download.html
```

## Local CLI Usage

All commands below are run from the `runpod/` directory.

### Pipeline 5 (BidirCrossTransformer) -- no text control

```bash
python main.py --npz samples/ex1.npz --wav samples/ex1.wav -o outputs/ex1_v5.mp4
```

This uses the default model (`bidir_cross_transformer`). To be explicit:

```bash
python main.py --npz samples/ex1.npz --wav samples/ex1.wav -o outputs/ex1_v5.mp4 --model v5
```

### Pipeline 6.1 (Emotion-Conditioned Transformer) -- text control + CFG

Basic run with a text prompt (no CFG, raw model output):

```bash
python main.py \
  --npz samples/ex1.npz \
  --wav samples/ex1.wav \
  -o outputs/ex1_happy.mp4 \
  --model v6.1 \
  --text-prompt "happy and cheerful speaker"
```

With Classifier-Free Guidance to amplify the emotion signal:

```bash
python main.py \
  --npz samples/ex1.npz \
  --wav samples/ex1.wav \
  -o outputs/ex1_angry_cfg5.mp4 \
  --model v6.1 \
  --text-prompt "angry and aggressive speaker" \
  --guidance-scale 5.0
```

Run v6.1 without text conditioning or FiLM modulation:

```bash
python main.py \
  --npz samples/ex1.npz \
  --wav samples/ex1.wav \
  -o outputs/ex1_v61_no_text.mp4 \
  --model v6.1 \
  --no-text-conditioning
```

**Guidance scale tips:**
- `1.0` or omitted: no amplification, raw model output
- `3.0`: moderate emotion steering (good default)
- `5.0`: strong emotion effect
- `7.0`: very strong -- may introduce artifacts on some clips

### Comparing different emotions on the same clip

```bash
python main.py --npz samples/ex1.npz --wav samples/ex1.wav -o outputs/ex1_sad.mp4 \
  --model v6.1 -t "sad and melancholic speaker" -g 5

python main.py --npz samples/ex1.npz --wav samples/ex1.wav -o outputs/ex1_excited.mp4 \
  --model v6.1 -t "excited and energetic speaker" -g 5

python main.py --npz samples/ex1.npz --wav samples/ex1.wav -o outputs/ex1_calm.mp4 \
  --model v6.1 -t "calm and relaxed speaker" -g 5
```

### Using v1 weights (Pipeline 6.0) explicitly

```bash
python main.py --npz samples/ex1.npz --wav samples/ex1.wav -o outputs/ex1_v1.mp4 \
  --model v6 --weights weights/best_emo_transformer.pt \
  --text-prompt "neutral speaker"
```

### Render options

Only show the predicted panel (faster):
```bash
python main.py --npz samples/ex1.npz --wav samples/ex1.wav -o outputs/pred_only.mp4 \
  --render-panels predicted
```

Higher quality render:
```bash
python main.py --npz samples/ex1.npz --wav samples/ex1.wav -o outputs/hq.mp4 \
  --image-size 384
```

## RunPod Serverless API

### Endpoint input

Send a RunPod request with `input`:

- `input_npz_uri` (**required**): `s3://...` or `https://...`
- `input_wav_uri` (**required**): `s3://...` or `https://...`
- `output_s3_uri` (**required**): output destination (`s3://bucket/key.mp4`)
- `model_name` (optional): see table above (default: `"bidir_cross_transformer"`)
- `weights_path` (optional): custom weights path inside container
- `text_prompt` (optional): free-form text control for `"emotion_conditioned_transformer"` (default: `"unknown emotion"`)
- `text_conditioning` (optional): set `false` for v6.1 to bypass text/FiLM modulation
- `guidance_scale` (optional): CFG scale for emotion amplification (default: none/disabled, try `3.0`-`7.0`)
- `image_size` (optional): panel size in pixels (default: `320`)
- `render_dist` (optional): camera distance (default: `0.78`)
- `bg_color` (optional): `[r, g, b]` floats in `[0,1]` (default: `[0.08, 0.08, 0.1]`)
- `render_scale` (optional): render super-sampling scale (default: `1.0`)
- `video_crf` (optional): H264 quality (default: `18`, lower = better, range `0..51`)
- `render_frame_stride` (optional): render every Nth frame (default: `1`)
- `render_panels` (optional): `"all"` or array of `"input"`, `"ground_truth"`, `"predicted"`

### Minimal request example

```json
{
  "input": {
    "input_npz_uri": "https://listencontrol.s3.us-east-1.amazonaws.com/ex1.npz",
    "input_wav_uri": "https://listencontrol.s3.us-east-1.amazonaws.com/ex1.wav",
    "output_s3_uri": "s3://listencontrol/outputs/result.mp4"
  }
}
```

### Pipeline 5 request (explicit v5)

```json
{
  "input": {
    "input_npz_uri": "s3://listencontrol/ex1.npz",
    "input_wav_uri": "s3://listencontrol/ex1.wav",
    "output_s3_uri": "s3://listencontrol/outputs/result_v5.mp4",
    "model_name": "v5",
    "weights_path": "weights/best_bidir_cross_transformer.pt",
    "image_size": 320,
    "render_panels": ["input", "ground_truth", "predicted"],
    "video_crf": 18
  }
}
```

### Pipeline 5 request (faster predicted-only render)

```json
{
  "input": {
    "input_npz_uri": "s3://listencontrol/ex1.npz",
    "input_wav_uri": "s3://listencontrol/ex1.wav",
    "output_s3_uri": "s3://listencontrol/outputs/result_v5_fast.mp4",
    "model_name": "v5",
    "weights_path": "weights/best_bidir_cross_transformer.pt",
    "image_size": 256,
    "render_panels": ["predicted"],
    "render_frame_stride": 2,
    "video_crf": 20
  }
}
```

### Text-controlled request with CFG (Pipeline 6.1)

```json
{
  "input": {
    "input_npz_uri": "s3://listencontrol/ex1.npz",
    "input_wav_uri": "s3://listencontrol/ex1.wav",
    "output_s3_uri": "s3://listencontrol/outputs/result_happy_cfg.mp4",
    "model_name": "v6.1",
    "text_prompt": "playful and cheerful speaker",
    "guidance_scale": 5.0,
    "image_size": 320,
    "render_panels": ["input", "ground_truth", "predicted"],
    "video_crf": 18
  }
}
```

### Pipeline 6.1 request without text modulation

This uses the v6.1 checkpoint but bypasses text/FiLM conditioning. Do not include `text_prompt` or `guidance_scale` for this mode.

```json
{
  "input": {
    "input_npz_uri": "s3://listencontrol/ex1.npz",
    "input_wav_uri": "s3://listencontrol/ex1.wav",
    "output_s3_uri": "s3://listencontrol/outputs/result_v61_no_text.mp4",
    "model_name": "v6.1",
    "text_conditioning": false,
    "image_size": 320,
    "render_panels": ["input", "ground_truth", "predicted"],
    "video_crf": 18
  }
}
```

### Text-controlled request with CFG + timeout policy (Pipeline 6.1)

Use this when full three-panel rendering needs more than the default RunPod execution timeout. `policy` is top-level, not inside `input`.

```json
{
  "input": {
    "input_npz_uri": "s3://listencontrol/ex1.npz",
    "input_wav_uri": "s3://listencontrol/ex1.wav",
    "output_s3_uri": "s3://listencontrol/outputs/result_happy_cfg_timeout.mp4",
    "model_name": "v6.1",
    "text_prompt": "playful and cheerful speaker",
    "guidance_scale": 5.0,
    "image_size": 320,
    "render_panels": ["input", "ground_truth", "predicted"],
    "video_crf": 18
  },
  "policy": {
    "executionTimeout": 1200000,
    "ttl": 1800000
  }
}
```

### Required environment variables (RunPod)

- `REGION_S3`
- `ACCESS_KEY_ID_S3`
- `SECRET_ACCESS_KEY_S3`

## Weight Files

Place these in `runpod/weights/`:

| File | Model |
|---|---|
| `best_bidir_cross_transformer.pt` | Pipeline 5 (v5) |
| `best_emo_transformer_v2.pt` | Pipeline 6.1 with CFG support |
| `best_emo_transformer.pt` | Pipeline 6.0 (v1 fallback) |
| `best_model_dim_128_30.pt` | Legacy 128 |
| `best_model_dim_256_30.pt` | Legacy 256 |

## Notes

- FLAME assets must be in `runpod/render/flame/`.
- The BidirCrossTransformer runs autoregressively at inference (`tf_ratio=0.0`).
- The emotion transformer encodes `text_prompt` with CLIP (`openai/clip-vit-base-patch32`) and applies FiLM modulation.
- Omitting `text_prompt` on v6.1 uses `"unknown emotion"`; set `text_conditioning: false` for no text/FiLM modulation.
- CFG runs the decoder twice (conditioned + unconditioned) and extrapolates: `y = y_uncond + scale * (y_cond - y_uncond)`.
- If `best_emo_transformer_v2.pt` is missing, the code automatically falls back to `best_emo_transformer.pt`.
- The Docker image uses `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` to avoid the `torch.load` CVE guard in `transformers`.
