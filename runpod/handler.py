"""
RunPod serverless handler: S3 download -> ListenControl pipeline -> S3 upload.
Expects input: input_npz_s3_uri, input_wav_s3_uri, output_s3_uri.
Uses env: REGION_S3, ACCESS_KEY_ID_S3, SECRET_ACCESS_KEY_S3.
"""
import os
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import runpod


def _validate_assets_on_startup():
    """Fail fast if required assets are missing."""
    root = Path(__file__).resolve().parent
    weights = Path(os.environ.get("LISTEN_WEIGHTS_PATH", str(root / "weights" / "best_model_dim_128_30.pt")))
    if not weights.exists():
        raise SystemExit(
            f"Missing ListenControl weights: {weights}\n"
            "Set LISTEN_WEIGHTS_PATH or add weights/best_model_dim_128_30.pt"
        )
    flame_dir = root / "render" / "flame"
    required = ["generic_model.pkl", "flame_static_embedding.pkl", "flame_dynamic_embedding.npy"]
    for name in required:
        p = flame_dir / name
        if not p.exists():
            raise SystemExit(f"Missing FLAME asset: {p}\nPlace FLAME files in {flame_dir}")


_validate_assets_on_startup()

# Lazy singleton predictor - initialized on first request
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from main import ListenControlPredictor

        root = Path(__file__).resolve().parent
        weights = os.environ.get("LISTEN_WEIGHTS_PATH", str(root / "weights" / "best_model_dim_128_30.pt"))
        _predictor = ListenControlPredictor(weights_path=weights, model_size="128")
    return _predictor


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


def _download_from_s3(client, uri: str, local_path: Path) -> None:
    bucket, key = _parse_s3_uri(uri)
    client.download_file(bucket, key, str(local_path))


def _upload_to_s3(client, local_path: Path, uri: str) -> None:
    bucket, key = _parse_s3_uri(uri)
    client.upload_file(str(local_path), bucket, key)


def handler(event):
    """
    Process RunPod request: download NPZ/WAV from S3, run pipeline, upload video.
    """
    job_id = event.get("id", "unknown")
    inp = event.get("input") or {}

    input_npz_uri = inp.get("input_npz_s3_uri")
    input_wav_uri = inp.get("input_wav_s3_uri")
    output_uri = inp.get("output_s3_uri")

    if not all([input_npz_uri, input_wav_uri, output_uri]):
        return {
            "status": "error",
            "error": "Missing required input: input_npz_s3_uri, input_wav_s3_uri, output_s3_uri",
        }

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
        # Download
        t0 = time.perf_counter()
        _download_from_s3(s3, input_npz_uri, npz_path)
        _download_from_s3(s3, input_wav_uri, wav_path)
        timings["download_sec"] = round(time.perf_counter() - t0, 2)
        print(f"[{job_id}] Downloaded in {timings['download_sec']}s")

        # Pipeline
        t1 = time.perf_counter()
        from main import run_pipeline

        predictor = _get_predictor()
        used_device = str(predictor.device)
        video_path = run_pipeline(
            npz_path=npz_path,
            wav_path=wav_path,
            output_path=out_path,
            predictor=predictor,
        )
        timings["pipeline_sec"] = round(time.perf_counter() - t1, 2)
        print(f"[{job_id}] Pipeline done in {timings['pipeline_sec']}s")

        # Upload (video_path may be output.mp4 or output_silent.mp4 if ffmpeg failed)
        t2 = time.perf_counter()
        _upload_to_s3(s3, Path(video_path), output_uri)
        timings["upload_sec"] = round(time.perf_counter() - t2, 2)
        print(f"[{job_id}] Uploaded in {timings['upload_sec']}s")

        total = round(time.perf_counter() - t0, 2)
        return {
            "status": "success",
            "output_s3_uri": output_uri,
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
