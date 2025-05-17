import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TransformerEncoder(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Improved attention with residual
        x_res = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x)
        x_attn = self.dropout(x_attn)
        x = x_attn + x_res

        # Improved FFN with residual
        x_res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * 0.8 + x_res * 0.2  # Weighted residual connection
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        # Generate spatial attention mask
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        mask = torch.sigmoid(self.conv(x_cat))
        
        # Apply attention mask
        return x * mask

class MobileViTBlock(nn.Module):
    def __init__(self, dim, ffn_dim, n_transformer_blocks=2, expansion_factor=4):
        super().__init__()
        # Improved local representation
        self.local_rep = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim * expansion_factor, kernel_size=1),
            nn.BatchNorm2d(dim * expansion_factor),
            nn.GELU(),
            nn.Conv2d(dim * expansion_factor, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerEncoder(dim, ffn_dim, num_heads=max(4, dim//64)) for _ in range(n_transformer_blocks)]
        )
        
        # Spatial attention for medical focus
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
    def forward(self, x):
        # Local representation
        x_local = self.local_rep(x)
        
        # Transformer processing
        B, C, H, W = x_local.shape
        x_flat = x_local.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_trans = self.transformer(x_flat)
        x_trans = x_trans.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Combine with residual and spatial attention
        x_combined = x + x_trans
        x_attended = self.spatial_attention(x_combined)
        
        return x_attended

class OptimizedConvNeXtMobileViT(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.2):
        super().__init__()
        # Load pretrained ConvNeXt as backbone
        self.backbone = timm.create_model("convnext_tiny", features_only=True, pretrained=False)
        
        # Get feature dimensions from backbone
        feature_dims = [96, 192, 384, 768]  # ConvNeXt tiny feature dimensions
        
        # Apply MobileViT blocks to different stages
        self.mobilevit1 = MobileViTBlock(dim=feature_dims[1], ffn_dim=feature_dims[1]*2, n_transformer_blocks=1)
        self.mobilevit2 = MobileViTBlock(dim=feature_dims[2], ffn_dim=feature_dims[2]*2, n_transformer_blocks=2)
        self.mobilevit3 = MobileViTBlock(dim=feature_dims[3], ffn_dim=feature_dims[3]*2, n_transformer_blocks=3)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Final classification head with dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(sum(feature_dims[1:]), num_classes)
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply MobileViT blocks to different stages
        x1 = self.mobilevit1(features[1])  # Stage 1
        x2 = self.mobilevit2(features[2])  # Stage 2
        x3 = self.mobilevit3(features[3])  # Stage 3
        
        # Global pooling
        p1 = self.pool(x1).flatten(1)
        p2 = self.pool(x2).flatten(1)
        p3 = self.pool(x3).flatten(1)
        
        # Combine multi-scale features
        combined = torch.cat([p1, p2, p3], dim=1)
        combined = self.dropout(combined)
        
        # Final prediction
        output = self.head(combined)
        
        return output

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal loss modification
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss