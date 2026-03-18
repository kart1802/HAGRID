import torch 
import torch.nn as nn
import torch.nn.functional as F
class DGGM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # QKV projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Learnable weights for geometry prior
        self.lambda1 = nn.Parameter(torch.tensor(1.0))  # depth weight
        self.lambda2 = nn.Parameter(torch.tensor(1.0))  # spatial weight

        # Decay factor η ∈ (0,1)
        self.eta = nn.Parameter(torch.tensor(0.9))

        # Cache for spatial distances
        self.spatial_cache = {}

    def get_spatial(self, H, W, device):
        """
        Compute Manhattan distance between all patch pairs
        ΔS ∈ (HW, HW)
        """
        key = (H, W)
        if key not in self.spatial_cache:
            coords = torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'),
                dim=-1
            ).reshape(-1, 2).to(device)  # (HW, 2)

            delta_S = torch.abs(coords.unsqueeze(1) - coords.unsqueeze(0)).sum(-1)
            self.spatial_cache[key] = delta_S

        return self.spatial_cache[key]

    def forward(self, x, depth):
        """
        x: (B, H, W, C)
        depth: (B, 1, Hd, Wd)
        """

        B, H, W, C = x.shape
        HW = H * W

        # ---------------------------------------
        # 1. Flatten features
        # ---------------------------------------
        x_flat = x.reshape(B, HW, C)  # (B, HW, C)

        # ---------------------------------------
        # 2. Q, K, V projections
        # ---------------------------------------
        Q = self.to_q(x_flat)
        K = self.to_k(x_flat)
        V = self.to_v(x_flat)

        # ---------------------------------------
        # 3. Standard attention
        # ---------------------------------------
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)  # (B, HW, HW)

        # ---------------------------------------
        # 4. Depth prior ΔD
        # ---------------------------------------
        # Resize depth to match feature resolution
        depth_resized = F.interpolate(
            depth, size=(H, W), mode='bilinear', align_corners=False
        )

        # Flatten depth → (B, HW)
        D = depth_resized.reshape(B, HW)

        # Pairwise depth difference → (B, HW, HW)
        delta_D = torch.abs(D.unsqueeze(2) - D.unsqueeze(1))

        # ---------------------------------------
        # 5. Spatial prior ΔS
        # ---------------------------------------
        delta_S = self.get_spatial(H, W, x.device)  # (HW, HW)
        delta_S = delta_S.unsqueeze(0)  # (1, HW, HW)

        # ---------------------------------------
        # 6. Geometry prior G
        # G = λ1 ΔD + λ2 ΔS
        # ---------------------------------------
        G = self.lambda1 * delta_D + self.lambda2 * delta_S

        # ---------------------------------------
        # 7. Apply η^G (paper-style)
        # ---------------------------------------
        geom_decay = torch.pow(self.eta, G)  # (B, HW, HW)

        # Apply AFTER softmax
        attn = attn * geom_decay

        # ---------------------------------------
        # 8. Final output
        # ---------------------------------------
        out = torch.matmul(attn, V)  # (B, HW, C)

        # Reshape back
        out = out.reshape(B, H, W, C)

        return out