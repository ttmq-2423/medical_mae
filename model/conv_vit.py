import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TransformerEncoder(nn.Module):
    def __init__(self, dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + x_res

        x_res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + x_res
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, ffn_dim, n_transformer_blocks=2):
        super().__init__()
        self.local_rep = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.transformer = nn.Sequential(
            *[TransformerEncoder(dim, ffn_dim) for _ in range(n_transformer_blocks)]
        )

    def forward(self, x):
        x = self.local_rep(x)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_trans = self.transformer(x_flat)
        x_trans = x_trans.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + x_trans  # residual connection

class ConvNeXtMobileViT(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", features_only=True, pretrained=False)
        # Stage 2 (feats[2]) has output channels = 384
        self.mobilevit = MobileViTBlock(dim=384, ffn_dim=768, n_transformer_blocks=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        x = feats[2]  # Use Stage 2
        x = self.mobilevit(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x  # for multi-label classification
