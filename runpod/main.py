from pathlib import Path
import shutil
import subprocess
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Model
from model.architecture import ListenControl128, ListenControl256
from render.render_pipeline import FlameRenderPipeline

DEFAULT_WEIGHTS_BY_SIZE = {
    "128": Path("weights/best_model_dim_128_30.pt"),
    "256": Path("weights/best_model_dim_256_30.pt"),
}


class ListenControlPredictor:
    """Loads all models once and runs flame + audio inference."""

    def __init__(
        self,
        weights_path=None,
        model_size="128",
        w2v_name="facebook/wav2vec2-base-960h",
        device=None,
    ):
        model_size = str(model_size)
        if model_size not in {"128", "256"}:
            raise ValueError(f"Invalid model_size={model_size}. Use '128' or '256'.")

        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_BY_SIZE[model_size]

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"ListenControl weights not found: {weights_path}\n"
                "Set weights_path or place model weights in the weights folder."
            )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w2v_model = Wav2Vec2Model.from_pretrained(w2v_name).to(self.device)
        self.w2v_model.eval()

        if model_size == "256":
            self.model = ListenControl256(flame_in_dim=56, out_dim=56).to(self.device)
        else:
            self.model = ListenControl128(flame_in_dim=56, out_dim=56).to(self.device)

        state_dict = torch.load(str(weights_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def batch_to_wav2vec_features(self, x_audio, x_lens, target_T=200):
        """
        Returns wav2vec features [B, target_T, C].
        """
        bsz, num_samples = x_audio.shape
        attn = (
            torch.arange(num_samples, device=x_audio.device).unsqueeze(0)
            < x_lens.unsqueeze(1)
        ).long()
        feats = self.w2v_model(input_values=x_audio, attention_mask=attn).last_hidden_state
        feats = feats.transpose(1, 2)
        feats = F.adaptive_avg_pool1d(feats, target_T)
        return feats.transpose(1, 2)

    def _load_audio(self, sample_path_audio):
        wav, sr = torchaudio.load(sample_path_audio)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0)

    @torch.no_grad()
    def predict(self, sample_path_flame, sample_path_audio):
        """
        Args:
            sample_path_flame: Path to .npz containing x_flame and y_flame.
            sample_path_audio: Path to .wav audio.
        Returns:
            x_flame, y_flame, predicted_flame (all numpy arrays with shape [T, 56]).
        """
        sample_path_flame = Path(sample_path_flame)
        sample_path_audio = Path(sample_path_audio)

        data = np.load(sample_path_flame)
        x_flame = data["x_flame"].astype(np.float32)
        y_flame = data["y_flame"].astype(np.float32)

        x_flame_tensor = torch.from_numpy(x_flame).unsqueeze(0).to(self.device)  # [1, T, 56]

        audio = self._load_audio(str(sample_path_audio)).to(self.device)  # [N]
        x_audio = audio.unsqueeze(0)  # [1, N]
        x_lens = torch.tensor([x_audio.shape[1]], device=self.device, dtype=torch.long)

        target_T = x_flame_tensor.shape[1]
        x_w2v = self.batch_to_wav2vec_features(x_audio, x_lens, target_T=target_T)  # [1, T, 768]
        predicted_flame = self.model(x_w2v, x_flame_tensor).squeeze(0).cpu().numpy()

        return x_flame, y_flame, predicted_flame


@torch.no_grad()
def save_comparison_video_with_audio(
    x_flame,
    y_flame,
    predicted_flame,
    audio_path,
    output_path="outputs/comparison_with_audio.mp4",
    shape_params=None,
    fps=25,
    expression_dim=50,
    pose_dim=6,
    image_size=320,
    render_dist=0.62,
    bg_color=(0.08, 0.08, 0.1),
):
    """
    Render original / ground-truth / predicted FLAME side-by-side video
    and attach audio track in the background.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_video_path = output_path.with_name(f"{output_path.stem}_silent.mp4")

    x_seq = np.asarray(x_flame, dtype=np.float32)
    y_seq = np.asarray(y_flame, dtype=np.float32)
    pred = np.asarray(predicted_flame, dtype=np.float32)
    if x_seq.ndim != 2 or y_seq.ndim != 2 or pred.ndim != 2:
        raise ValueError("x_flame, y_flame, predicted_flame must be [T, D] arrays.")

    def init_renderer(device_override=None):
        local_renderer = FlameRenderPipeline(device=device_override)
        if shape_params is None:
            shape_dim = int(getattr(local_renderer.config, "shape_params", 100))
            local_shape = torch.zeros((1, shape_dim), device=local_renderer.device)
        else:
            local_shape = local_renderer._to_tensor(shape_params)
        return local_renderer, local_shape

    num_frames = min(x_seq.shape[0], y_seq.shape[0], pred.shape[0])
    renderer, shape_tensor = init_renderer()

    def render_one_frame(frame_vec):
        frame_vec = torch.from_numpy(frame_vec).to(renderer.device)
        exp, pose = renderer.split_flame_vector(
            frame_vec,
            expression_dim=expression_dim,
            pose_dim=pose_dim,
        )
        vertices, _ = renderer.forward(
            shape_params=shape_tensor,
            expression_params=exp,
            pose_params=pose,
        )
        image = renderer.render_vertices(
            vertices,
            image_size=image_size,
            dist=render_dist,
            bg_color=bg_color,
        )[0]
        return (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # On some Windows setups, PyTorch3D installs without CUDA kernels.
    # If that happens, transparently fall back to CPU rendering.
    if renderer.device.type == "cuda":
        try:
            _ = render_one_frame(x_seq[0])
        except RuntimeError as e:
            if "Not compiled with GPU support" in str(e):
                print("PyTorch3D GPU support unavailable. Falling back to CPU rendering.")
                renderer, shape_tensor = init_renderer(torch.device("cpu"))
            else:
                raise

    ffmpeg_path = shutil.which("ffmpeg")

    def iter_combined_frames():
        for frame_idx in range(num_frames):
            frame_x = render_one_frame(x_seq[frame_idx])
            frame_y = render_one_frame(y_seq[frame_idx])
            frame_pred = render_one_frame(pred[frame_idx])
            yield np.concatenate([frame_x, frame_y, frame_pred], axis=1)

    # Prefer imageio writer. Some worker environments miss the plugin backend.
    # Fall back to direct ffmpeg piping so jobs still succeed.
    try:
        with imageio.get_writer(str(silent_video_path), fps=fps) as writer:
            for combined in iter_combined_frames():
                writer.append_data(combined)
    except ValueError as e:
        message = str(e)
        if "Could not find a backend" not in message:
            raise
        if ffmpeg_path is None:
            raise RuntimeError(
                "Failed to write MP4: imageio backend missing and ffmpeg binary not found."
            ) from e
        print("imageio MP4 backend missing. Falling back to direct ffmpeg encoding.")
        width = int(image_size * 3)
        height = int(image_size)
        encode_cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(silent_video_path),
        ]
        proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert proc.stdin is not None
            for combined in iter_combined_frames():
                proc.stdin.write(combined.tobytes())
            proc.stdin.close()
            _, stderr_data = proc.communicate()
        except Exception:
            proc.kill()
            raise
        if proc.returncode != 0:
            err_text = stderr_data.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg failed to encode video: {err_text}")

    if ffmpeg_path is None:
        print("ffmpeg not found. Returning silent comparison video.")
        return silent_video_path

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(silent_video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print("Failed to merge audio with video. Returning silent video.")
        if e.stderr:
            print(e.stderr)
        return silent_video_path


def run_pipeline(
    npz_path,
    wav_path,
    output_path,
    predictor=None,
    weights_path=None,
    model_size="128",
    fps=25,
    image_size=320,
    render_dist=0.62,
    bg_color=(0.08, 0.08, 0.1),
):
    """
    Single entrypoint for serverless: predict + render comparison video.
    Returns path to final MP4 (with audio if ffmpeg available).
    """
    npz_path = Path(npz_path)
    wav_path = Path(wav_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if predictor is None:
        predictor = ListenControlPredictor(
            weights_path=weights_path,
            model_size=model_size,
        )

    x_flame, y_flame, predicted_flame = predictor.predict(
        sample_path_flame=npz_path,
        sample_path_audio=wav_path,
    )

    shape_params = None
    with np.load(npz_path) as data:
        if "shape" in data:
            shape_params = data["shape"][0]

    video_path = save_comparison_video_with_audio(
        x_flame=x_flame,
        y_flame=y_flame,
        predicted_flame=predicted_flame,
        audio_path=wav_path,
        output_path=output_path,
        shape_params=shape_params,
        fps=fps,
        image_size=image_size,
        render_dist=render_dist,
        bg_color=bg_color,
    )
    return video_path


if __name__ == "__main__":
    sample_path_flame = "samples/ex1.npz"
    sample_path_audio = "samples/ex1.wav"
    video_path = run_pipeline(
        npz_path=sample_path_flame,
        wav_path=sample_path_audio,
        output_path="outputs/comparison_with_audio.mp4",
    )
    print("saved_video:", str(video_path))
