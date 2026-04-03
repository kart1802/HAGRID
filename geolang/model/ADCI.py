import torch
import torch.nn as nn
import torch.nn.functional as F

class ADCI(nn.Module):
    def __init__(self, in_channels_list, align_channels, out_channels, L, G, target_size):
        """
        Adaptive Dense Channel Integration (ADCI)
        
        Args:
            in_channels_list: List of channel dimensions for the L input features.
            align_channels: The common channel dimension to project features to before summing.
            out_channels: The final output channel dimension (C_v).
            L: Total number of input layers/features.
            G: Number of groups. L must be divisible by G.
            target_size: Tuple (H, W) for the target spatial resolution (e.g., H/16, W/16).
        """
        super(ADCI, self).__init__()
        assert L % G == 0, "Number of layers L must be exactly divisible by number of groups G."
        
        self.L = L
        self.G = G
        self.M = L // G
        self.target_size = target_size
        
        # 1. Feature Alignment Networks
        # Projects each multi-scale feature to the same channel dimension so they can be summed.
        self.align_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, align_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(align_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in in_channels_list
        ])
        
        # 2. Gating Networks (Step 2 & 3)
        # Calculates the adaptive unnormalized score 's_i' for each feature.
        reduction_ratio = 4
        self.gating_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, max(in_channels // reduction_ratio, 16)),
                nn.ReLU(inplace=True),
                nn.Linear(max(in_channels // reduction_ratio, 16), 1)
            ) for in_channels in in_channels_list
        ])
        
        # 3. Final Multi-Layer Perceptron (Step 6)
        # Maps the concatenated group features and the last layer to the final embedding.
        # Concatenation size = (G * align_channels) + align_channels (for C_L)
        concat_channels = (G + 1) * align_channels
        self.final_mlp = nn.Sequential(
            nn.Conv2d(concat_channels, concat_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(concat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(concat_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, features):
        assert len(features) == self.L, f"Expected {self.L} features, got {len(features)}"
        B = features[0].size(0)
        
        aligned_features = []
        scores = []
        
        # Process each feature map
        for i in range(self.L):
            C_i = features[i]
            
            # --- Steps 2 & 3: Gating Network ---
            # z_i = GAP(C_i)
            z_i = F.adaptive_avg_pool2d(C_i, 1).view(B, -1) 
            
            # s_i = W_2 * sigma(W_1 * z_i + b_1) + b_2
            s_i = self.gating_mlps[i](z_i) # Shape: (B, 1)

            scores.append(s_i)
            
            # --- Spatial and Channel Alignment ---
            # Project channels and interpolate to target spatial resolution (H/16, W/16)
            aligned_C_i = self.align_convs[i](C_i)
            
            if aligned_C_i.shape[-2:] != self.target_size:
                aligned_C_i = F.interpolate(aligned_C_i, size=self.target_size, mode='bilinear', align_corners=False)
            aligned_features.append(aligned_C_i)
        
        group_features = []
        
        # --- Steps 1, 4, 5: Grouping and Feature Aggregation ---
        for g in range(self.G):
            start_idx = g * self.M
            end_idx = start_idx + self.M
            
            # Extract scores and features for the current group
            group_scores = torch.cat(scores[start_idx:end_idx], dim=1) # Shape: (B, M)
            
            # alpha_i = Softmax(s_i) within the group
            group_alphas = F.softmax(group_scores, dim=1) # Shape: (B, M)
            
            # GC_g = \sum \alpha_i C_i
            # Initialize accumulator for the group feature
            GC_g = torch.zeros_like(aligned_features[start_idx]) 
            for j in range(self.M):
                # Reshape alpha for broadcasting: (B, 1, 1, 1)
                alpha_ij = group_alphas[:, j].view(B, 1, 1, 1)
                GC_g = GC_g + alpha_ij * aligned_features[start_idx + j]
                
            group_features.append(GC_g)
        
        # --- Step 6: Final Embedding ---
        # Concatenate group features with the aligned last layer C_L
        C_L = aligned_features[-1]
        concat_feature = torch.cat(group_features + [C_L], dim=1)
        
        # e_v = MLP(Concatenate([GC_1, ..., GC_G, C_L]))
        e_v = self.final_mlp(concat_feature)
        
        return e_v


def test_adci():
    print("Testing ADCI module based on GeoLanG paper...")
    
    # 1. Mocking the GeoLanG multi-scale setup
    B = 2
    # Downsampling factors for layer_1, layer_2, layer_3 (excluding layer_0)
    # Assume an input image of 416x416. 
    # H/16 = 26, H/32 = 13.
    target_size = (26, 26) # Matches the H/16, W/16 target resolution from the paper
    
    C_list = [512, 1024, 1024]    # Number of channels from VMamba stages (layer_1, layer_2, layer_3)
    H_list = [26, 13, 13]         # Spatial heights
    W_list = [26, 13, 13]         # Spatial widths
    
    L = 3                         # 3 layers
    G = 1                         # 1 group containing all 3 layers
    align_channels = 256          # Unified channel dim for summation
    out_channels = 512            # Final multimodal visual embedding size
    
    # 2. Generate mock multi-scale feature maps from DGGM
    features = []
    for i in range(L):
        features.append(torch.randn(B, C_list[i], H_list[i], W_list[i]))
    
    #print(features[0].shape, features[1].shape, features[2].shape)
        
    print(f"Input features shapes (C1, C2, C3):")
    for f in features:
        print(f"  {tuple(f.shape)}")
        
    # 3. Initialize ADCI
    adci = ADCI(
        in_channels_list=C_list,
        align_channels=align_channels,
        out_channels=out_channels,
        L=L,
        G=G,
        target_size=target_size
    )
    
    # 4. Forward Pass
    e_v = adci(features)
    
    print(f"\nTarget spatial resolution: {target_size} (H/16, W/16)")
    print(f"Output embedding shape: {tuple(e_v.shape)}")
    
    # Assert output shape is strictly (B, final_channels, target_H, target_W)
    assert e_v.shape == (B, out_channels, target_size[0], target_size[1]), f"Unexpected shape {e_v.shape}"
    print("-> Forward pass successful! Output is a dense spatial map.")
    
    # 5. Backward Pass
    loss = e_v.sum()
    loss.backward()
    
    # Ensure gradients are flowing back to all multi-scale inputs
    for i, f in enumerate(features):
        # features are created without requires_grad by default in torch.randn,
        # so let's verify parameters in the module received gradients instead.
        pass
    
    # Check if a parameter in the gating network received a gradient
    assert adci.gating_mlps[0][0].weight.grad is not None, "Gradients did not flow through the gating network!"
    assert adci.final_mlp[0].weight.grad is not None, "Gradients did not flow through the final MLP!"
    print("-> Backward pass successful! Gradients are flowing correctly.")

if __name__ == "__main__":
    test_adci()