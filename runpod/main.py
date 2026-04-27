from pathlib import Path
import shutil
import subprocess
import time
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import CLIPTextModel, CLIPTokenizer, Wav2Vec2Model
from model.architecture import (
    BidirCrossTransformer,
    EmotionConditionedTransformer,
    ListenControl128,
    ListenControl256,
)
from render.render_pipeline import FlameRenderPipeline

DEFAULT_MODEL_NAME = "bidir_cross_transformer"
DEFAULT_TEXT_PROMPT = "unknown emotion"
FLAME_VECTOR_DIM = 56
DEFAULT_TARGET_FRAMES = 200

DEFAULT_WEIGHTS_BY_MODEL = {
    "bidir_cross_transformer": Path("weights/best_bidir_cross_transformer.pt"),
    "emotion_conditioned_transformer": Path("weights/best_emo_transformer_v2.pt"),
    "128": Path("weights/best_model_dim_128_30.pt"),
    "256": Path("weights/best_model_dim_256_30.pt"),
}

FALLBACK_WEIGHTS = {
    "emotion_conditioned_transformer": Path("weights/best_emo_transformer.pt"),
}

DEFAULT_RENDER_PANELS = ("input", "ground_truth", "predicted")


def load_torch_checkpoint(weights_path, device):
    try:
        return torch.load(str(weights_path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(weights_path), map_location=device)


def normalize_model_name(model_name=None, model_size=None):
    value = model_name if model_name is not None else model_size
    if value is None:
        return DEFAULT_MODEL_NAME

    value = str(value).strip().lower()
    aliases = {
        "bidir": "bidir_cross_transformer",
        "bidir_cross": "bidir_cross_transformer",
        "bidir_cross_transformer": "bidir_cross_transformer",
        "bidircrosstransformer": "bidir_cross_transformer",
        "cross_transformer": "bidir_cross_transformer",
        "transformer": "bidir_cross_transformer",
        "v5": "bidir_cross_transformer",
        "emotion_conditioned_transformer": "emotion_conditioned_transformer",
        "emotion_transformer": "emotion_conditioned_transformer",
        "emo_transformer": "emotion_conditioned_transformer",
        "text_control_transformer": "emotion_conditioned_transformer",
        "text_conditioned_transformer": "emotion_conditioned_transformer",
        "p6": "emotion_conditioned_transformer",
        "v6": "emotion_conditioned_transformer",
        "v6.1": "emotion_conditioned_transformer",
        "128": "128",
        "256": "256",
    }
    if value not in aliases:
        allowed = "', '".join(DEFAULT_WEIGHTS_BY_MODEL)
        raise ValueError(f"Invalid model_name={value}. Use one of: '{allowed}'.")
    return aliases[value]


def normalize_render_panels(render_panels=None):
    if render_panels is None:
        return DEFAULT_RENDER_PANELS

    if isinstance(render_panels, str):
        value = render_panels.strip().lower()
        if value in {"all", "comparison", "default"}:
            return DEFAULT_RENDER_PANELS
        panel_values = [p.strip() for p in value.replace("|", ",").split(",") if p.strip()]
    else:
        panel_values = list(render_panels)

    aliases = {
        "input": "input",
        "original": "input",
        "source": "input",
        "x": "input",
        "ground_truth": "ground_truth",
        "groundtruth": "ground_truth",
        "target": "ground_truth",
        "true": "ground_truth",
        "y": "ground_truth",
        "predicted": "predicted",
        "prediction": "predicted",
        "pred": "predicted",
    }
    panels = []
    for panel in panel_values:
        key = str(panel).strip().lower()
        if key not in aliases:
            allowed = "', '".join(["input", "ground_truth", "predicted"])
            raise ValueError(f"Invalid render panel={panel}. Use one or more of: '{allowed}'.")
        normalized = aliases[key]
        if normalized not in panels:
            panels.append(normalized)

    if not panels:
        raise ValueError("render_panels must include at least one panel.")
    return tuple(panels)


def normalize_flame_mode(flame_mode=None):
    value = str(flame_mode or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "strict": "strict",
        "required": "strict",
        "require": "strict",
        "zeros": "zeros",
        "zero": "zeros",
        "none": "zeros",
        "missing": "zeros",
        "audio_only": "zeros",
        "audio-only": "zeros",
        "no_flame": "zeros",
        "no-flame": "zeros",
    }
    if value not in aliases:
        allowed = "', '".join(["auto", "strict", "zeros"])
        raise ValueError(f"Invalid flame_mode={flame_mode}. Use one of: '{allowed}'.")
    return aliases[value]


def _validate_target_frames(target_frames=None):
    if target_frames is None:
        return DEFAULT_TARGET_FRAMES
    target_frames = int(target_frames)
    if target_frames < 1 or target_frames > DEFAULT_TARGET_FRAMES:
        raise ValueError(
            f"target_frames must be between 1 and {DEFAULT_TARGET_FRAMES}; "
            "the current transformer checkpoints were trained with max_len=200."
        )
    return target_frames


def _as_flame_sequence(name, value):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != FLAME_VECTOR_DIM:
        raise ValueError(f"{name} must have shape [T, {FLAME_VECTOR_DIM}], got {arr.shape}.")
    return arr


def _zero_flame_sequence(num_frames):
    return np.zeros((int(num_frames), FLAME_VECTOR_DIM), dtype=np.float32)


def load_flame_sequences(sample_path_flame=None, target_frames=None, flame_mode="auto"):
    """Load FLAME conditioning/target arrays, or create zero input for audio-only inference."""
    mode = normalize_flame_mode(flame_mode)
    fallback_frames = _validate_target_frames(target_frames)

    if sample_path_flame is None:
        if mode == "strict":
            raise ValueError("input_npz_uri/--npz is required when flame_mode='strict'.")
        x_flame = _zero_flame_sequence(fallback_frames)
        return x_flame, x_flame.copy()

    sample_path_flame = Path(sample_path_flame)
    if not sample_path_flame.exists():
        raise FileNotFoundError(f"FLAME npz not found: {sample_path_flame}")

    x_flame = None
    y_flame = None
    with np.load(sample_path_flame) as data:
        if "x_flame" in data:
            x_flame = _as_flame_sequence("x_flame", data["x_flame"])
        if "y_flame" in data:
            y_flame = _as_flame_sequence("y_flame", data["y_flame"])

    if x_flame is None:
        if mode == "strict":
            raise KeyError(f"{sample_path_flame} is missing required key 'x_flame'.")
        frame_count = y_flame.shape[0] if y_flame is not None else fallback_frames
        x_flame = _zero_flame_sequence(frame_count)

    if y_flame is None:
        y_flame = _zero_flame_sequence(x_flame.shape[0])

    return x_flame, y_flame


class ListenControlPredictor:
    """Loads all models once and runs flame + audio inference."""

    def __init__(
        self,
        weights_path=None,
        model_size=None,
        model_name=None,
        w2v_name="facebook/wav2vec2-base-960h",
        clip_name="openai/clip-vit-base-patch32",
        device=None,
    ):
        self.model_name = normalize_model_name(model_name=model_name, model_size=model_size)

        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_BY_MODEL[self.model_name]
            if not Path(weights_path).exists() and self.model_name in FALLBACK_WEIGHTS:
                weights_path = FALLBACK_WEIGHTS[self.model_name]

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"ListenControl weights not found: {weights_path}\n"
                "Set weights_path or place model weights in the weights folder."
            )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w2v_model = Wav2Vec2Model.from_pretrained(w2v_name).to(self.device)
        self.w2v_model.eval()
        self.clip_tokenizer = None
        self.clip_text_model = None
        self._uncond_emb = None

        if self.model_name == "bidir_cross_transformer":
            self.model = BidirCrossTransformer(
                w2v_dim=768,
                flame_in_dim=56,
                d_model=256,
                nhead=8,
                num_layers=3,
                ff_dim=1024,
                out_dim=56,
                dropout=0.2,
                max_len=200,
                gru_hidden=512,
                gru_layers=2,
            ).to(self.device)
        elif self.model_name == "emotion_conditioned_transformer":
            self.model = EmotionConditionedTransformer(
                w2v_dim=768,
                flame_in_dim=56,
                d_model=256,
                nhead=8,
                num_layers=3,
                ff_dim=1024,
                out_dim=56,
                dropout=0.2,
                max_len=200,
                gru_hidden=512,
                gru_layers=2,
                clip_dim=512,
                emo_dim=64,
            ).to(self.device)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_name)
            self.clip_text_model = CLIPTextModel.from_pretrained(clip_name).to(self.device)
            self.clip_text_model.eval()
        elif self.model_name == "256":
            self.model = ListenControl256(flame_in_dim=56, out_dim=56).to(self.device)
        else:
            self.model = ListenControl128(flame_in_dim=56, out_dim=56).to(self.device)

        checkpoint = load_torch_checkpoint(weights_path, self.device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if self.model_name == "emotion_conditioned_transformer":
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            unexpected = [k for k in incompatible.unexpected_keys if k != "clip_emb"]
            if incompatible.missing_keys or unexpected:
                raise RuntimeError(
                    "Failed to load EmotionConditionedTransformer weights. "
                    f"missing={incompatible.missing_keys}, unexpected={unexpected}"
                )
        else:
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
    def encode_text_prompt(self, text_prompt=None):
        if self.clip_tokenizer is None or self.clip_text_model is None:
            raise ValueError("text_prompt is only supported by the emotion_conditioned_transformer model.")

        text_prompt = str(text_prompt or DEFAULT_TEXT_PROMPT).strip() or DEFAULT_TEXT_PROMPT
        tokens = self.clip_tokenizer(
            [text_prompt[:77]],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        emb = self.clip_text_model(**tokens).pooler_output
        return emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def get_uncond_emb(self):
        """Cached CLIP embedding for 'unknown emotion' (used as CFG baseline)."""
        if self._uncond_emb is None:
            self._uncond_emb = self.encode_text_prompt("unknown emotion")
        return self._uncond_emb

    @torch.no_grad()
    def predict(self, sample_path_flame, sample_path_audio, text_prompt=None,
                guidance_scale=None, text_conditioning=True, flame_mode="auto",
                target_frames=None):
        """
        Args:
            sample_path_flame: Path to .npz containing x_flame and y_flame, or
                None for audio-only inference with zero FLAME conditioning.
            sample_path_audio: Path to .wav audio.
            text_prompt: Optional free-form emotion/control prompt for pipeline 6.
            guidance_scale: If > 1.0, uses Classifier-Free Guidance to amplify
                the emotion signal. Recommended: 3.0-7.0.
            text_conditioning: If False for pipeline 6, bypasses FiLM text modulation.
            flame_mode: "auto"/"zeros" allows missing FLAME by using zeros;
                "strict" requires x_flame in the NPZ.
            target_frames: Number of output frames when no FLAME sequence is provided.
        Returns:
            x_flame, y_flame, predicted_flame (all numpy arrays with shape [T, 56]).
        """
        sample_path_flame = Path(sample_path_flame) if sample_path_flame is not None else None
        sample_path_audio = Path(sample_path_audio)

        x_flame, y_flame = load_flame_sequences(
            sample_path_flame=sample_path_flame,
            target_frames=target_frames,
            flame_mode=flame_mode,
        )

        x_flame_tensor = torch.from_numpy(x_flame).unsqueeze(0).to(self.device)  # [1, T, 56]

        audio = self._load_audio(str(sample_path_audio)).to(self.device)  # [N]
        x_audio = audio.unsqueeze(0)  # [1, N]
        x_lens = torch.tensor([x_audio.shape[1]], device=self.device, dtype=torch.long)

        target_T = x_flame_tensor.shape[1]
        if hasattr(self.model, "pos_emb") and target_T > self.model.pos_emb.shape[1]:
            raise ValueError(
                f"target sequence has {target_T} frames, but this checkpoint supports "
                f"at most {self.model.pos_emb.shape[1]} frames."
            )
        x_w2v = self.batch_to_wav2vec_features(x_audio, x_lens, target_T=target_T)  # [1, T, 768]

        if self.model_name == "emotion_conditioned_transformer":
            if not text_conditioning:
                context = self.model.encode(x_w2v, x_flame_tensor)
                predicted_flame = self.model.decode_ar(
                    context,
                    y_gt=None,
                    tf_ratio=0.0,
                ).squeeze(0).cpu().numpy()
                return x_flame, y_flame, predicted_flame

            text_emb = self.encode_text_prompt(text_prompt).expand(x_w2v.shape[0], -1)

            if guidance_scale is not None and guidance_scale > 1.0:
                uncond_emb = self.get_uncond_emb().expand(x_w2v.shape[0], -1)
                predicted_flame = self.model.forward_cfg(
                    x_w2v,
                    x_flame_tensor,
                    text_emb=text_emb,
                    uncond_text_emb=uncond_emb,
                    guidance_scale=guidance_scale,
                ).squeeze(0).cpu().numpy()
            else:
                predicted_flame = self.model(
                    x_w2v,
                    x_flame_tensor,
                    text_emb=text_emb,
                    y_gt=None,
                    tf_ratio=0.0,
                ).squeeze(0).cpu().numpy()
        elif self.model_name == "bidir_cross_transformer":
            predicted_flame = self.model(
                x_w2v,
                x_flame_tensor,
                y_gt=None,
                tf_ratio=0.0,
            ).squeeze(0).cpu().numpy()
        else:
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
    render_dist=0.78,
    bg_color=(0.08, 0.08, 0.1),
    render_scale=1.0,
    video_crf=18,
    render_frame_stride=1,
    render_panels=None,
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
    image_size = int(image_size)
    if image_size < 64:
        raise ValueError("image_size must be >= 64.")
    render_scale = float(render_scale)
    if render_scale < 1.0:
        raise ValueError("render_scale must be >= 1.0.")
    video_crf = int(video_crf)
    if video_crf < 0 or video_crf > 51:
        raise ValueError("video_crf must be between 0 and 51 (lower means better quality).")
    render_frame_stride = int(render_frame_stride)
    if render_frame_stride < 1:
        raise ValueError("render_frame_stride must be >= 1.")
    panels = normalize_render_panels(render_panels)
    render_image_size = int(round(image_size * render_scale))

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
    source_frame_count = (num_frames + render_frame_stride - 1) // render_frame_stride
    if render_frame_stride > 1 or panels != DEFAULT_RENDER_PANELS:
        print(
            "Rendering "
            f"{source_frame_count}/{num_frames} unique frames, panels={','.join(panels)}"
        )

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
            image_size=render_image_size,
            dist=render_dist,
            bg_color=bg_color,
        )[0]
        if render_image_size != image_size:
            image_chw = image.permute(2, 0, 1).unsqueeze(0)
            try:
                image = F.interpolate(
                    image_chw,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0].permute(1, 2, 0)
            except TypeError:
                image = F.interpolate(
                    image_chw,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )[0].permute(1, 2, 0)
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
    panel_sequences = {
        "input": x_seq,
        "ground_truth": y_seq,
        "predicted": pred,
    }

    def iter_combined_frames():
        for frame_idx in range(0, num_frames, render_frame_stride):
            rendered_panels = [
                render_one_frame(panel_sequences[panel][frame_idx])
                for panel in panels
            ]
            if len(rendered_panels) == 1:
                combined = rendered_panels[0]
            else:
                combined = np.concatenate(rendered_panels, axis=1)
            repeat_count = min(render_frame_stride, num_frames - frame_idx)
            for _ in range(repeat_count):
                yield combined

    # Prefer imageio writer. Some worker environments miss the plugin backend.
    # Fall back to direct ffmpeg piping so jobs still succeed.
    def write_with_imageio(use_quality_settings):
        writer_kwargs = {"fps": fps}
        if use_quality_settings:
            writer_kwargs["codec"] = "libx264"
            writer_kwargs["ffmpeg_params"] = ["-crf", str(video_crf), "-preset", "medium"]
        with imageio.get_writer(str(silent_video_path), **writer_kwargs) as writer:
            for combined in iter_combined_frames():
                writer.append_data(combined)

    try:
        try:
            write_with_imageio(use_quality_settings=True)
        except TypeError:
            # Some backends do not accept ffmpeg-specific kwargs.
            write_with_imageio(use_quality_settings=False)
    except ValueError as e:
        message = str(e)
        if "Could not find a backend" not in message:
            raise
        if ffmpeg_path is None:
            raise RuntimeError(
                "Failed to write MP4: imageio backend missing and ffmpeg binary not found."
            ) from e
        print("imageio MP4 backend missing. Falling back to direct ffmpeg encoding.")
        width = int(image_size * len(panels))
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
            "-crf",
            str(video_crf),
            "-preset",
            "medium",
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
    model_size=None,
    model_name=None,
    fps=25,
    image_size=320,
    render_dist=0.78,
    bg_color=(0.08, 0.08, 0.1),
    render_scale=1.0,
    video_crf=18,
    render_frame_stride=1,
    render_panels=None,
    text_prompt=None,
    guidance_scale=None,
    text_conditioning=True,
    flame_mode="auto",
    target_frames=None,
    timings=None,
):
    """
    Single entrypoint for serverless: predict + render comparison video.
    Returns path to final MP4 (with audio if ffmpeg available).
    """
    npz_path = Path(npz_path) if npz_path is not None else None
    wav_path = Path(wav_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if predictor is None:
        predictor = ListenControlPredictor(
            weights_path=weights_path,
            model_name=model_name,
            model_size=model_size,
        )

    if timings is None:
        timings = {}

    t_predict = time.perf_counter()
    x_flame, y_flame, predicted_flame = predictor.predict(
        sample_path_flame=npz_path,
        sample_path_audio=wav_path,
        text_prompt=text_prompt,
        guidance_scale=guidance_scale,
        text_conditioning=text_conditioning,
        flame_mode=flame_mode,
        target_frames=target_frames,
    )
    timings["predict_sec"] = round(time.perf_counter() - t_predict, 2)

    shape_params = None
    if npz_path is not None and npz_path.exists():
        with np.load(npz_path) as data:
            if "shape" in data:
                shape_params = data["shape"][0]

    t_render = time.perf_counter()
    if npz_path is None and render_panels is None:
        render_panels = ("predicted",)
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
        render_scale=render_scale,
        video_crf=video_crf,
        render_frame_stride=render_frame_stride,
        render_panels=render_panels,
    )
    timings["render_sec"] = round(time.perf_counter() - t_render, 2)
    return video_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ListenControl local inference CLI")
    parser.add_argument("--npz", default="samples/ex1.npz", help="Path to .npz file")
    parser.add_argument("--wav", default="samples/ex1.wav", help="Path to .wav file")
    parser.add_argument("--output", "-o", default="outputs/comparison_with_audio.mp4")
    parser.add_argument("--model", default=None,
                        help="Model name: bidir_cross_transformer, emotion_conditioned_transformer, 128, 256")
    parser.add_argument("--weights", default=None, help="Path to .pt weights file")
    parser.add_argument("--text-prompt", "-t", default=None,
                        help="Emotion/control prompt (emo_transformer only)")
    parser.add_argument("--guidance-scale", "-g", type=float, default=None,
                        help="CFG guidance scale (emo_transformer only, try 3-7)")
    parser.add_argument("--no-text-conditioning", action="store_true",
                        help="For v6/v6.1, bypass text/FiLM modulation.")
    parser.add_argument("--no-flame", action="store_true",
                        help="Run audio-only by feeding zero FLAME conditioning.")
    parser.add_argument("--flame-mode", default="auto", choices=["auto", "strict", "zeros"],
                        help="How to handle missing x_flame in the NPZ.")
    parser.add_argument("--target-frames", type=int, default=None,
                        help="Output frame count when --no-flame is used (default: 200).")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--render-panels", nargs="*", default=None,
                        help="Panels to render: input ground_truth predicted")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    npz_path = None if args.no_flame else args.npz
    flame_mode = "zeros" if args.no_flame else args.flame_mode

    video_path = run_pipeline(
        npz_path=npz_path,
        wav_path=args.wav,
        output_path=args.output,
        model_name=args.model,
        weights_path=args.weights,
        text_prompt=args.text_prompt,
        guidance_scale=args.guidance_scale,
        text_conditioning=not args.no_text_conditioning,
        flame_mode=flame_mode,
        target_frames=args.target_frames,
        image_size=args.image_size,
        render_panels=args.render_panels,
        fps=args.fps,
    )
    print("saved_video:", str(video_path))
