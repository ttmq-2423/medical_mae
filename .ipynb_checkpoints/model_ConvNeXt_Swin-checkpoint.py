import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
from timm.models.convnext import ConvNeXt

class ConvNeXt_Swin(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Stage 1, 2, 3: ConvNeXt Blocks
        self.convnext = ConvNeXt(
            depths=(3, 3, 27, 3),  
            dims=(96, 192, 384, 768)  
        )

        # 🔧 Chuyển đổi số kênh từ 384 ➝ 3 để phù hợp với Swin Transformer
        self.channel_reduction = nn.Conv2d(384, 3, kernel_size=1, stride=1, padding=0)

        # Stage 4: Swin Transformer
        self.swin_stage4 = SwinTransformer(
            img_size=7,   
            embed_dim=3,  # 🔧 Để khớp với số kênh đầu vào
            depths=[3],    
            num_heads=[3]  
        )

        # Global Pooling & Fully Connected Layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(3, num_classes)  # 🔧 Đầu vào FC phải khớp với embed_dim

    def forward(self, x):
        x = self.convnext.stem(x)  
        x = self.convnext.stages[0](x)  
        x = self.convnext.stages[1](x)  
        x = self.convnext.stages[2](x)  

        print("Shape before Swin Transformer:", x.shape)  # Debug shape

        x = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))  # ✅ Chuẩn hóa kích thước
        x = self.channel_reduction(x)  # ✅ Chuyển từ 384 → 3 kênh
        x = self.swin_stage4(x)  

        print("Shape before Global Pooling:", x.shape)  # Debug shape

        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)  

        print("Shape before FC:", x.shape)  # Debug shape

        x = self.fc(x)  
        return x
