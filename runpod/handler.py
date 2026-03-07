"""
RunPod handler: input download (S3 or HTTPS) -> inference/render -> S3 upload.
Supports selecting model size/weights and render sizing per request.
"""
import os
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import runpod


def _normalize_model_size(model_size) -> str:
    model_size = str(model_size or "128").strip()
    if model_size not in {"128", "256"}:
        raise ValueError(f"Invalid model_size={model_size}. Use '128' or '256'.")
    return model_size


def _resolve_weights_path(root: Path, model_size: str, explicit_weights_path=None) -> Path:
    if explicit_weights_path:
        p = Path(explicit_weights_path)
        if not p.is_absolute():
            p = root / p
        return p

    env_specific = os.environ.get(f"LISTEN_WEIGHTS_PATH_{model_size}")
    if env_specific:
        return Path(env_specific)

    env_general = os.environ.get("LISTEN_WEIGHTS_PATH")
    if env_general:
        return Path(env_general)

    default_name = f"best_model_dim_{model_size}_30.pt"
    return root / "weights" / default_name


def _validate_assets_on_startup():
    """Fail fast if required FLAME assets and at least one weight file exist."""
    root = Path(__file__).resolve().parent
    flame_dir = root / "render" / "flame"
    required = ["generic_model.pkl", "flame_static_embedding.pkl", "flame_dynamic_embedding.npy"]
    for name in required:
        p = flame_dir / name
        if not p.exists():
            raise SystemExit(f"Missing FLAME asset: {p}\nPlace FLAME files in {flame_dir}")

    weights_dir = root / "weights"
    has_env_weight = bool(os.environ.get("LISTEN_WEIGHTS_PATH") or os.environ.get("LISTEN_WEIGHTS_PATH_128") or os.environ.get("LISTEN_WEIGHTS_PATH_256"))
    has_local_weight = weights_dir.exists() and any(weights_dir.glob("*.pt"))
    if not has_env_weight and not has_local_weight:
        raise SystemExit(
            f"No model weights found in {weights_dir} and no LISTEN_WEIGHTS_PATH envs set.\n"
            "Add at least one .pt file or set LISTEN_WEIGHTS_PATH(_128/_256)."
        )


_validate_assets_on_startup()

# Lazy predictor cache keyed by (model_size, weights_path)
_predictors = {}


def _get_predictor(model_size: str, weights_path: Path):
    key = (model_size, str(weights_path.resolve()))
    predictor = _predictors.get(key)
    if predictor is not None:
        return predictor

    from main import ListenControlPredictor

    predictor = ListenControlPredictor(weights_path=str(weights_path), model_size=model_size)
    _predictors[key] = predictor
    return predictor


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def _download_input(client, uri: str, local_path: Path) -> None:
    """Download from s3:// or http(s):// URI to local file path."""
    uri = (uri or "").strip()
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(uri)
        client.download_file(bucket, key, str(local_path))
        return
    if uri.startswith("http://") or uri.startswith("https://"):
        urlretrieve(uri, str(local_path))
        return
    raise ValueError(f"Unsupported input URI: {uri}. Use s3:// or http(s)://")


def _upload_to_s3(client, local_path: Path, uri: str) -> None:
    bucket, key = _parse_s3_uri(uri)
    client.upload_file(str(local_path), bucket, key)


def _parse_bg_color(bg_value):
    if bg_value is None:
        return (0.08, 0.08, 0.1)
    if not isinstance(bg_value, (list, tuple)) or len(bg_value) != 3:
        raise ValueError("bg_color must be an array with 3 values, e.g. [0.08, 0.08, 0.1]")
    return tuple(float(v) for v in bg_value)


def handler(event):
    """
    Process request:
    - Download NPZ/WAV from S3 or HTTPS.
    - Run inference + render.
    - Upload output video to output_s3_uri.
    """
    job_id = event.get("id", "unknown")
    inp = event.get("input") or {}

    input_npz_uri = inp.get("input_npz_uri") or inp.get("input_npz_s3_uri")
    input_wav_uri = inp.get("input_wav_uri") or inp.get("input_wav_s3_uri")
    output_uri = inp.get("output_s3_uri")

    if not all([input_npz_uri, input_wav_uri, output_uri]):
        return {
            "status": "error",
            "error": "Missing required input: input_npz_uri, input_wav_uri, output_s3_uri",
        }

    model_size = _normalize_model_size(inp.get("model_size", "128"))
    render_image_size = int(inp.get("image_size", 320))
    render_dist = float(inp.get("render_dist", 0.62))
    bg_color = _parse_bg_color(inp.get("bg_color"))

    region = os.environ.get("REGION_S3", "us-east-1")
    access_key = os.environ.get("ACCESS_KEY_ID_S3")
    secret_key = os.environ.get("SECRET_ACCESS_KEY_S3")
    if not access_key or not secret_key:
        return {"status": "error", "error": "S3 credentials not configured (ACCESS_KEY_ID_S3, SECRET_ACCESS_KEY_S3)"}

    import boto3

    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    work_dir = Path(tempfile.mkdtemp(prefix=f"runpod_{job_id}_"))
    npz_path = work_dir / "input.npz"
    wav_path = work_dir / "input.wav"
    out_path = work_dir / "output.mp4"

    timings = {}
    used_device = "cpu"

    try:
        root = Path(__file__).resolve().parent
        weights_path = _resolve_weights_path(root, model_size, explicit_weights_path=inp.get("weights_path"))
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Requested weights file does not exist: {weights_path}\n"
                "Set weights_path in input or LISTEN_WEIGHTS_PATH(_128/_256)."
            )

        t0 = time.perf_counter()
        _download_input(s3, input_npz_uri, npz_path)
        _download_input(s3, input_wav_uri, wav_path)
        timings["download_sec"] = round(time.perf_counter() - t0, 2)
        print(f"[{job_id}] Downloaded in {timings['download_sec']}s")

        t1 = time.perf_counter()
        from main import run_pipeline

        predictor = _get_predictor(model_size=model_size, weights_path=weights_path)
        used_device = str(predictor.device)
        video_path = run_pipeline(
            npz_path=npz_path,
            wav_path=wav_path,
            output_path=out_path,
            predictor=predictor,
            model_size=model_size,
            image_size=render_image_size,
            render_dist=render_dist,
            bg_color=bg_color,
        )
        timings["pipeline_sec"] = round(time.perf_counter() - t1, 2)
        print(f"[{job_id}] Pipeline done in {timings['pipeline_sec']}s")

        t2 = time.perf_counter()
        _upload_to_s3(s3, Path(video_path), output_uri)
        timings["upload_sec"] = round(time.perf_counter() - t2, 2)
        print(f"[{job_id}] Uploaded in {timings['upload_sec']}s")

        total = round(time.perf_counter() - t0, 2)
        return {
            "status": "success",
            "output_s3_uri": output_uri,
            "model_size": model_size,
            "weights_path": str(weights_path),
            "image_size": render_image_size,
            "render_dist": render_dist,
            "bg_color": bg_color,
            "duration_sec": total,
            "timings": timings,
            "used_device": used_device,
        }

    except ValueError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        print(f"[{job_id}] Error: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"[{job_id}] Cleaned up {work_dir}")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
