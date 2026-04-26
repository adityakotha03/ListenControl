import torch
import torch.nn as nn


class ListenControl128(nn.Module):
    def __init__(
        self,
        w2v_dim=768,
        flame_in_dim=56,
        hidden=256,
        num_layers=2,
        out_dim=56,
        dropout=0.1,
        attn_dim=128,
        num_heads=4,
    ):
        super().__init__()

        # 768 -> 128
        self.w2v_proj = nn.Linear(w2v_dim, attn_dim)

        # 56 -> 64 -> 128
        self.flame_proj = nn.Sequential(
            nn.Linear(flame_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, attn_dim),
            nn.ReLU(),
        )

        self.cross_attn_f2w = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn_w2f = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm_w = nn.LayerNorm(attn_dim)
        self.norm_f = nn.LayerNorm(attn_dim)

        # Concatenating [f2w, w2f] => 128 + 128 = 256
        lstm_in_dim = attn_dim * 2  # 256

        self.lstm = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x_w2v, x_flame):
        """
        x_w2v:   [B, T, 768]
        x_flame: [B, T, 56]
        """
        w = self.w2v_proj(x_w2v)      # [B, T, 128]
        f = self.flame_proj(x_flame)  # [B, T, 128]

        f2w, _ = self.cross_attn_f2w(query=f, key=w, value=w)  # [B, T, 128]
        f2w = self.norm_f(f + f2w)

        w2f, _ = self.cross_attn_w2f(query=w, key=f, value=f)  # [B, T, 128]
        w2f = self.norm_w(w + w2f)

        x = torch.cat([f2w, w2f], dim=-1)  # [B, T, 256]
        h, _ = self.lstm(x)                # [B, T, hidden]
        out = self.proj(h)                 # [B, T, out_dim=56]
        return out
    
    

class ListenControl256(nn.Module):
    def __init__(
        self,
        w2v_dim=768,
        flame_in_dim=56,
        hidden=512,
        num_layers=2,
        out_dim=56,
        dropout=0.1,
        attn_dim=256,
        num_heads=8,
    ):
        super().__init__()

        # 768 -> 256
        self.w2v_proj = nn.Linear(w2v_dim, attn_dim)

        # 56 -> 128 -> 256
        self.flame_proj = nn.Sequential(
            nn.Linear(flame_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, attn_dim),
            nn.ReLU(),
        )

        self.cross_attn_f2w = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn_w2f = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm_w = nn.LayerNorm(attn_dim)
        self.norm_f = nn.LayerNorm(attn_dim)

        # Concatenating [f2w, w2f] => 256 + 256 = 512
        lstm_in_dim = attn_dim * 2  # 512

        self.lstm = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x_w2v, x_flame):
        """
        x_w2v:   [B, T, 768]
        x_flame: [B, T, 56]
        """
        w = self.w2v_proj(x_w2v)      # [B, T, 256]
        f = self.flame_proj(x_flame)  # [B, T, 256]

        f2w, _ = self.cross_attn_f2w(query=f, key=w, value=w)  # [B, T, 256]
        f2w = self.norm_f(f + f2w)

        w2f, _ = self.cross_attn_w2f(query=w, key=f, value=f)  # [B, T, 256]
        w2f = self.norm_w(w + w2f)

        x = torch.cat([f2w, w2f], dim=-1)  # [B, T, 512]
        h, _ = self.lstm(x)                # [B, T, hidden]
        out = self.proj(h)                 # [B, T, out_dim=56]
        return out

class BidirCrossTransformer(nn.Module):
    def __init__(
        self,
        w2v_dim=768,
        flame_in_dim=56,
        d_model=256,
        nhead=8,
        num_layers=3,
        ff_dim=1024,
        out_dim=56,
        dropout=0.1,
        max_len=200,
        gru_hidden=512,
        gru_layers=2,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.d_model = d_model

        # --- Audio pathway ---
        self.audio_in_proj = nn.Linear(w2v_dim, d_model)

        self.audio_local = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- FLAME pathway ---
        self.visual_proj = nn.Sequential(
            nn.Linear(flame_in_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )

        # --- Bidirectional cross-attention ---
        self.cross_attn_a2v = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.cross_attn_v2a = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

        # --- Fusion projection ---
        self.fuse_proj = nn.Linear(d_model * 2, d_model)

        # --- Positional embedding + Transformer encoder ---
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(d_model)

        # --- Autoregressive GRU decoder ---
        self.gru = nn.GRU(
            input_size=d_model + out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.out_head = nn.Linear(gru_hidden, out_dim)

    def encode(self, x_w2v, x_flame):
        """Encoder: audio + FLAME -> context sequence [B, T, d_model]."""
        a = self.audio_in_proj(x_w2v)
        a = a + self.audio_local(a.transpose(1, 2)).transpose(1, 2)

        v = self.visual_proj(x_flame)

        a2v, _ = self.cross_attn_a2v(query=a, key=v, value=v)
        a_fused = self.norm_a(a + a2v)

        v2a, _ = self.cross_attn_v2a(query=v, key=a, value=a)
        v_fused = self.norm_v(v + v2a)

        x = torch.cat([a_fused, v_fused], dim=-1)
        x = self.fuse_proj(x)

        T = x.size(1)
        x = x + self.pos_emb[:, :T]

        x = self.encoder(x)
        x = self.enc_norm(x)
        return x

    def decode_ar(self, context, y_gt=None, tf_ratio=1.0):
        """
        Autoregressive GRU decoder with fast paths.
        context: [B, T, d_model] from encoder.
        y_gt: [B, T, out_dim] ground truth for teacher forcing (None at inference).
        tf_ratio: probability of using ground truth at each step during training.
        """
        B, T, _ = context.shape
        device = context.device

        # FAST PATH: full teacher forcing -- single batched GRU call, no loop
        if y_gt is not None and tf_ratio >= 1.0:
            zeros = torch.zeros(B, 1, self.out_dim, device=device)
            prev_gt = torch.cat([zeros, y_gt[:, :-1, :]], dim=1)  # [B, T, out_dim]
            gru_in = torch.cat([context, prev_gt], dim=-1)        # [B, T, d_model+out_dim]
            out, _ = self.gru(gru_in)                             # [B, T, gru_hidden]
            return self.out_head(out)                              # [B, T, out_dim]

        # SLOW PATH: partial teacher forcing or pure autoregressive
        prev = torch.zeros(B, 1, self.out_dim, device=device)
        h = None
        outputs = []

        for t in range(T):
            ctx_t = context[:, t:t+1, :]
            gru_in = torch.cat([ctx_t, prev], dim=-1)
            out_t, h = self.gru(gru_in, h)
            pred_t = self.out_head(out_t)
            outputs.append(pred_t)

            if y_gt is not None and torch.rand(1).item() < tf_ratio:
                prev = y_gt[:, t:t+1, :]
            else:
                prev = pred_t.detach()

        return torch.cat(outputs, dim=1)

    def forward(self, x_w2v, x_flame, y_gt=None, tf_ratio=1.0):
        context = self.encode(x_w2v, x_flame)
        return self.decode_ar(context, y_gt=y_gt, tf_ratio=tf_ratio)


class EmotionConditionedTransformer(BidirCrossTransformer):
    """BidirCrossTransformer with CLIP text-conditioning via FiLM."""

    def __init__(self, *args, clip_emb_matrix=None, clip_dim=512, emo_dim=64, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_emb_matrix is not None:
            self.register_buffer("clip_emb", clip_emb_matrix)
            clip_dim = clip_emb_matrix.shape[-1]
        self.emo_proj = nn.Sequential(
            nn.Linear(clip_dim, emo_dim),
            nn.GELU(),
            nn.Linear(emo_dim, emo_dim),
        )
        self.film_gamma = nn.Linear(emo_dim, self.d_model)
        self.film_beta = nn.Linear(emo_dim, self.d_model)

    def _emo_features(self, emo_idx=None, text_emb=None):
        if text_emb is not None:
            return self.emo_proj(text_emb.to(self.film_gamma.weight.device))
        if emo_idx is None:
            raise ValueError("EmotionConditionedTransformer requires text_emb or emo_idx.")
        if not hasattr(self, "clip_emb"):
            raise ValueError("emo_idx inference requires a loaded clip_emb buffer.")
        return self.emo_proj(self.clip_emb[emo_idx])

    def _apply_film(self, context, emo_idx=None, text_emb=None):
        """Apply FiLM modulation to encoder context."""
        e = self._emo_features(emo_idx=emo_idx, text_emb=text_emb)
        gamma = self.film_gamma(e).unsqueeze(1)
        beta = self.film_beta(e).unsqueeze(1)
        return context * (1.0 + gamma) + beta

    def forward(self, x_w2v, x_flame, emo_idx=None, y_gt=None, tf_ratio=1.0, text_emb=None):
        context = self.encode(x_w2v, x_flame)
        context = self._apply_film(context, emo_idx=emo_idx, text_emb=text_emb)
        return self.decode_ar(context, y_gt=y_gt, tf_ratio=tf_ratio)

    @torch.no_grad()
    def forward_cfg(self, x_w2v, x_flame, emo_idx=None, text_emb=None,
                    uncond_text_emb=None, guidance_scale=3.0):
        """Classifier-Free Guidance inference.

        Runs the encoder once, then applies FiLM twice (conditioned +
        unconditioned) and extrapolates:
          y_guided = y_uncond + guidance_scale * (y_cond - y_uncond)
        """
        context = self.encode(x_w2v, x_flame)

        ctx_cond = self._apply_film(context, emo_idx=emo_idx, text_emb=text_emb)
        y_cond = self.decode_ar(ctx_cond, y_gt=None, tf_ratio=0.0)

        if uncond_text_emb is None:
            raise ValueError("forward_cfg requires uncond_text_emb (CLIP embedding for 'unknown emotion').")
        ctx_uncond = self._apply_film(context, text_emb=uncond_text_emb)
        y_uncond = self.decode_ar(ctx_uncond, y_gt=None, tf_ratio=0.0)

        return y_uncond + guidance_scale * (y_cond - y_uncond)
