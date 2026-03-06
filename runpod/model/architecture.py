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