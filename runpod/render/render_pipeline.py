import copy
import inspect
import sys
from pathlib import Path
import numpy as np
import torch


def _apply_compat_patches():
    """Compatibility patches for older FLAME codepaths."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    if not hasattr(np, "int"):
        np.int = int
        np.bool = bool
        np.float = float
        np.complex = complex
        np.object = object
        np.unicode = str
        np.str = str


class FlameRenderPipeline:
    """
    Loads FLAME once and renders meshes from FLAME params.

    Notes:
    - Expects FLAME files in `runpod/render/flame`.
    """

    def __init__(self, flame_dir=None, device=None, batch_size=1):
        _apply_compat_patches()

        # Ensure local FLAME_PyTorch repo is importable.
        runpod_root = Path(__file__).resolve().parent.parent
        repo_path = runpod_root / "FLAME_PyTorch"
        repo_path_str = str(repo_path)
        if repo_path.exists() and repo_path_str not in sys.path:
            sys.path.append(repo_path_str)

        from flame_pytorch import FLAME, get_config     # type: ignore[import-not-found]

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flame_dir = (
            Path(flame_dir)
            if flame_dir is not None
            else Path(__file__).resolve().parent / "flame"
        )
        self.flame_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.flame_dir / "generic_model.pkl"
        self.static_path = self.flame_dir / "flame_static_embedding.pkl"
        self.dynamic_path = self.flame_dir / "flame_dynamic_embedding.npy"

        self._ensure_local_flame_assets()

        original_argv = copy.copy(sys.argv)
        try:
            sys.argv = [""]
            config = get_config()
        finally:
            sys.argv = original_argv

        config.flame_model_path = str(self.model_path)
        config.static_landmark_embedding_path = str(self.static_path)
        config.dynamic_landmark_embedding_path = str(self.dynamic_path)
        config.batch_size = batch_size
        self.config = config

        self.flame_model = FLAME(self.config).to(self.device)
        self.flame_model.eval()

    def _is_valid_file(self, path, min_size_bytes):
        return path.exists() and path.stat().st_size >= min_size_bytes

    def _ensure_local_flame_assets(self):
        assets = [
            (self.model_path, 100_000),
            (self.static_path, 1_000),
            (self.dynamic_path, 1_000),
        ]
        missing_or_invalid = [
            str(path) for path, min_size in assets if not self._is_valid_file(path, min_size)
        ]
        if missing_or_invalid:
            joined = "\n".join(missing_or_invalid)
            raise FileNotFoundError(
                "Missing or invalid FLAME files in local flame folder:\n"
                f"{joined}\n"
                "Place these files inside runpod/render/flame."
            )

    def _to_tensor(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def forward(self, shape_params, expression_params, pose_params):
        shape_tensor = self._to_tensor(shape_params)
        exp_tensor = self._to_tensor(expression_params)
        pose_tensor = self._to_tensor(pose_params)
        vertices, landmarks = self.flame_model(
            shape_params=shape_tensor,
            expression_params=exp_tensor,
            pose_params=pose_tensor,
        )
        return vertices, landmarks

    def split_flame_vector(self, flame_vector, expression_dim=50, pose_dim=6):
        """
        Split your 56-dim vectors (or compatible) into expression and pose.
        Typical setup: expression_dim=50 and pose_dim=6.
        """
        flame_vector = self._to_tensor(flame_vector)
        needed_dim = expression_dim + pose_dim
        if flame_vector.shape[-1] < needed_dim:
            raise ValueError(
                f"Expected at least {needed_dim} dims, got {flame_vector.shape[-1]}."
            )
        exp = flame_vector[..., :expression_dim]
        pose = flame_vector[..., expression_dim : expression_dim + pose_dim]
        return exp, pose

    @torch.no_grad()
    def render_vertices(
        self,
        vertices,
        image_size=1024,
        dist=0.62,
        elev=0.0,
        azim=0.0,
        skin_color=(0.8, 0.6, 0.5),
        bg_color=(0.08, 0.08, 0.1),
    ):
        from pytorch3d.renderer import (  # type: ignore[import-not-found]
            BlendParams,
            FoVPerspectiveCameras,
            Materials,
            MeshRasterizer,
            MeshRenderer,
            PointLights,
            RasterizationSettings,
            SoftPhongShader,
            TexturesVertex,
            look_at_view_transform,
        )
        from pytorch3d.structures import Meshes  # type: ignore[import-not-found]

        vertices = self._to_tensor(vertices)
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)

        faces = (
            torch.tensor(self.flame_model.faces, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(vertices.shape[0], -1, -1)
        )
        base_color = torch.tensor([skin_color], device=self.device, dtype=torch.float32)
        verts_rgb = base_color.expand(vertices.shape[0], vertices.shape[1], 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        meshes = Meshes(verts=vertices, faces=faces, textures=textures)

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        lights = PointLights(device=self.device, location=[[0.0, 1.0, 3.0]])
        materials = Materials(
            device=self.device,
            specular_color=[[0.3, 0.3, 0.3]],
            shininess=15.0,
        )
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                materials=materials,
                blend_params=BlendParams(background_color=bg_color),
            ),
        )
        image = renderer(meshes)[..., :3]
        return image

    @torch.no_grad()
    def render_from_main_output(
        self,
        predicted_flame,
        shape_params=None,
        frame_idx=0,
        expression_dim=50,
        pose_dim=6,
        show=True,
    ):
        """
        Renders one frame from `predicted_flame` returned by main.py.

        Args:
            predicted_flame: [T, 56] or [56]
            shape_params: optional FLAME shape params; if None uses zeros.
            frame_idx: frame index when predicted_flame is [T, D]
        """
        pred = self._to_tensor(predicted_flame)
        if pred.dim() == 2:
            pred = pred[frame_idx]

        exp, pose = self.split_flame_vector(
            pred,
            expression_dim=expression_dim,
            pose_dim=pose_dim,
        )

        if shape_params is None:
            shape_dim = int(getattr(self.config, "shape_params", 100))
            shape_params = torch.zeros((1, shape_dim), device=self.device)

        vertices, landmarks = self.forward(
            shape_params=shape_params,
            expression_params=exp,
            pose_params=pose,
        )
        image = self.render_vertices(vertices)

        if show:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))
            plt.imshow(image[0].detach().cpu().numpy())
            plt.axis("off")
            plt.title("FLAME Render")
            plt.show()

        return {
            "vertices": vertices.detach().cpu(),
            "landmarks": landmarks.detach().cpu(),
            "image": image.detach().cpu(),
        }
