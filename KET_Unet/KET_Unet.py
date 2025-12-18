import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange




class Residual_AAG(nn.Module):
    """
    Residual Atomic-Scale Attention Gate (AAG)
    Output = x + alpha * AAG(g, x)
    """
    def __init__(self, F_g, F_l, F_int, filter_size=5):
        super().__init__()
        self.eps = 1e-6

        # ---------- local mean filter μ(x,y) ----------
        self.avg_filter = nn.Conv2d(
            F_l, F_l,
            kernel_size=filter_size,
            stride=1,
            padding=filter_size // 2,
            bias=False,
            groups=F_l
        )
        self.avg_filter.weight.data.fill_(1.0 / (filter_size ** 2))
        self.avg_filter.requires_grad_(False)

        # ---------- learnable spatial suppression λ(x,y) ----------
        self.lambda_conv = nn.Conv2d(F_l, 1, kernel_size=5, padding=2)

        # ---------- attention gate (same spirit as your original) ----------
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        # ---------- residual strength (learnable) ----------
        self.alpha = nn.Parameter(torch.zeros(1))  # start from "no AAG"

    def forward(self, g, x):
        """
        g: decoder feature (B, F_g, H, W)
        x: skip feature    (B, F_l, H, W)
        """

        # ===== 1. local normalization =====
        mu = self.avg_filter(x)
        mu = mu + self.eps
        N = x / mu                        # relative brightness

        # ===== 2. global normalization =====
        x_norm = (x - x.mean(dim=(2,3), keepdim=True)) / \
                 (x.std(dim=(2,3), keepdim=True) + self.eps)

        # ===== 3. learnable suppression =====
        lam = self.lambda_conv(x_norm)    # (B,1,H,W)
        x_suppressed = x - lam * N        # Eq.(16) style

        # ===== 4. attention gate =====
        g1 = self.W_g(g)
        x1 = self.W_x(x_suppressed)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        gated = x_suppressed * psi

        # ===== 5. residual bypass =====
        out = x + self.alpha * gated

        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim):
        super().__init__()
        # 关键：batch_first=True，输入输出都是 (B, N, C)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, dim),
        )

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, N, C)
        """
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)

        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x



class KET_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=512, num_heads=8, ff_dim=2048):
        super().__init__()
        # -------- Encoder --------
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, embed_dim)
        self.bottleneck = self.conv_block(embed_dim, embed_dim)

        # -------- Transformer bottleneck --------
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)

        # -------- Decoder (upsample only) --------
        self.decoder4 = self.upconv_block(embed_dim, 256)
        self.decoder3 = self.upconv_block(256, 128)
        self.decoder2 = self.upconv_block(128, 64)
        self.decoder1 = self.upconv_block(64, 64)

        # -------- After-concat channel fix (keep your original sizes) --------
        # dec4(256) + enc4(embed_dim=512) = 768
        self.conv_after_cat4 = nn.Conv2d(256 + embed_dim, 256, kernel_size=1)
        # dec3(128) + enc3(256) = 384
        self.conv_after_cat3 = nn.Conv2d(128 + 256, 128, kernel_size=1)
        # dec2(64) + enc2(128) = 192
        self.conv_after_cat2 = nn.Conv2d(64 + 128, 64, kernel_size=1)
        # dec1(64) + enc1(64) = 128
        self.conv_after_cat1 = nn.Conv2d(64 + 64, 64, kernel_size=1)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # -------- AAG on first two skip connections (paper-consistent) --------
        # enc2 gate: g is decoder feature at same scale (dec2:64ch), x is enc2 (128ch)
        self.aag2 = Residual_AAG(F_g=64, F_l=128, F_int=64)
        # enc1 gate: g is dec1 (64ch), x is enc1 (64ch)
        self.aag1 = Residual_AAG(F_g=64, F_l=64, F_int=32)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # -------- Encoder --------
        enc1 = self.encoder1(x)                                  # (B,64,256,256)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))   # (B,128,128,128)
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))   # (B,256,64,64)
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))   # (B,512,32,32)
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))       # (B,512,16,16)

        # -------- Transformer bottleneck --------
        B, C, H, W = bottleneck.shape
        z = rearrange(bottleneck, 'b c h w -> b (h w) c')         # (B,256,512)
        z = self.transformer(z)                                   # (B,256,512)
        bottleneck = rearrange(z, 'b (h w) c -> b c h w', h=H, w=W)

        # -------- Decoder stage 4 (16->32) --------
        dec4 = self.decoder4(bottleneck)                          # (B,256,32,32)
        dec4 = self.conv_after_cat4(torch.cat([dec4, enc4], dim=1))  # (B,256,32,32)

        # -------- Decoder stage 3 (32->64) --------
        dec3 = self.decoder3(dec4)                                # (B,128,64,64)
        dec3 = self.conv_after_cat3(torch.cat([dec3, enc3], dim=1))  # (B,128,64,64)

        # -------- Decoder stage 2 (64->128) + AAG on enc2 skip --------
        dec2 = self.decoder2(dec3)                                # (B,64,128,128)
        enc2_gated = self.aag2(g=dec2, x=enc2)                    # (B,128,128,128)
        dec2 = self.conv_after_cat2(torch.cat([dec2, enc2_gated], dim=1))  # (B,64,128,128)

        # -------- Decoder stage 1 (128->256) + AAG on enc1 skip --------
        dec1 = self.decoder1(dec2)                                # (B,64,256,256)
        enc1_gated = self.aag1(g=dec1, x=enc1)                    # (B,64,256,256)
        dec1 = self.conv_after_cat1(torch.cat([dec1, enc1_gated], dim=1))  # (B,64,256,256)

        output = self.final_conv(dec1)                            # (B,out_channels,256,256)
        return output



# 示例用法
if __name__ == "__main__":
    model = KET_UNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 256, 256)  # 示例输入
    output = model(x)
    print(output.shape)  # 输出形状
