# ListenControl RunPod API

Serverless pipeline that:
1. downloads `npz` + `wav` inputs,
2. runs ListenControl inference,
3. renders `original | ground_truth | predicted` FLAME video,
4. uploads final `mp4` to S3.

## Endpoint input

Send a RunPod request with `input`:

- `input_npz_uri` (**required**): `s3://...` or `https://...`
- `input_wav_uri` (**required**): `s3://...` or `https://...`
- `output_s3_uri` (**required**): output destination (`s3://bucket/key.mp4`)
- `model_size` (optional): `"128"` or `"256"` (default: `"128"`)
- `weights_path` (optional): custom weights path inside container
- `image_size` (optional): panel size in pixels (default: `320`)
- `render_dist` (optional): camera distance (default: `0.78`, lower = larger face, higher = more zoomed out)
- `bg_color` (optional): `[r, g, b]` floats in `[0,1]` (default: `[0.08, 0.08, 0.1]`)

Backward-compatible input keys still accepted:
- `input_npz_s3_uri`
- `input_wav_s3_uri`

## Minimal request example

```json
{
  "input": {
    "input_npz_uri": "https://listencontrol.s3.us-east-1.amazonaws.com/ex1.npz",
    "input_wav_uri": "https://listencontrol.s3.us-east-1.amazonaws.com/ex1.wav",
    "output_s3_uri": "s3://listencontrol/outputs/result.mp4"
  }
}
```

## Full request example (256 model + render tuning)

```json
{
  "input": {
    "input_npz_uri": "s3://listencontrol/ex1.npz",
    "input_wav_uri": "s3://listencontrol/ex1.wav",
    "output_s3_uri": "s3://listencontrol/outputs/result_256.mp4",
    "model_size": "256",
    "weights_path": "weights/best_model_dim_256_15.pt",
    "image_size": 256,
    "render_dist": 0.78,
    "bg_color": [0.08, 0.08, 0.1]
  }
}
```

## Success response example

```json
{
  "status": "success",
  "output_s3_uri": "s3://listencontrol/outputs/result_256.mp4",
  "model_size": "256",
  "weights_path": "/app/weights/best_model_dim_256_15.pt",
  "image_size": 256,
  "render_dist": 0.78,
  "bg_color": [0.08, 0.08, 0.1],
  "duration_sec": 42.73,
  "timings": {
    "download_sec": 1.24,
    "pipeline_sec": 38.97,
    "upload_sec": 2.52
  },
  "used_device": "cuda:0"
}
```

## Required environment variables (RunPod)

- `REGION_S3`
- `ACCESS_KEY_ID_S3`
- `SECRET_ACCESS_KEY_S3`

Optional weights envs:
- `LISTEN_WEIGHTS_PATH` (global default)
- `LISTEN_WEIGHTS_PATH_128` (default for model 128)
- `LISTEN_WEIGHTS_PATH_256` (default for model 256)

## Notes

- FLAME assets must be present in `runpod/render/flame/`.
- `weights_path` can be absolute (for example `/app/weights/...`) or relative to `/app`.
- If `imageio` mp4 backend is unavailable, the pipeline falls back to direct `ffmpeg` encoding.