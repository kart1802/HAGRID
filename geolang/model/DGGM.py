import torch 
import torch.nn as nn
import torch.nn.functional as F
class DGGM(nn.Module):
    def __init__(self, dim):
        
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.lambda1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.eta = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))
        
        
        self.spatial_cache = {}
        
    def get_spatial(self,H,W,device):
        key = (H,W)
        if key not in self.spatial_cache:
            coords = torch.stack(torch.meshgrid(torch.arange(H), 
                                torch.arange(W),indexing='ij'), dim=-1).reshape(-1, 2).to(device)
            
            delta = torch.abs(coords.unsqueeze(1) - coords.unsqueeze(0)).sum(-1)
            self.spatial_cache[key] = delta
            
        return self.spatial_cache[key]
    
    def forward(self,x,depth):
        """
        Args:
            x (_type_): _description_
            depth (_type_): _description_
        """
        
        B,H,W,C = x.shape
        HW = H*W
        
        x_flat = x.reshape(B, HW, C)
        
        Q = self.to_q(x_flat)
        K = self.to_k(x_flat)
        V = self.to_v(x_flat)
        
        attn = torch.matmul(Q, K.transpose(-2, -1) / (C ** 0.5))
        attn = torch.softmax(attn, dim=-1)
        
        depth_resized = F.interpolate(depth,size = (H,W), mode ='bilinear') 
        D = depth_resized.flatten(2).squeeze(1) # B, HW
        
        delta_D = torch.abs(D.unsqueeze(2).squeeze(1))
        
        delta_S = self.get_spatial(H,W,x.device) # HW, HW
        delta_S = delta_S.unsqueeze(0)
        
        G = self.lambda1 * delta_D + self.lambda2 * delta_S
        
        geom_deca = torch.pow(self.eta,G)
        
        attn = attn * geom_deca
        
        out = torch.matmul(attn, V) # B, HW, C
        
        out = out.reshape(B,H,W,C)
        
        
        return out 
    
    
    
        
    class MultiScaleDGGM(nn.Module):
        def __init__(self):
            super().__init__()

            self.dggm_0 = DGGM(256)
            self.dggm_1 = DGGM(512)
            self.dggm_2 = DGGM(1024)
            # self.dggm_3 = DGGM(1024)

        def forward(self, features, depth):
            """
            features: list of 4 tensors
            """

            f0, f1, f2, f3 = features

            f0 = self.dggm_0(f0, depth)
            f1 = self.dggm_1(f1, depth)
            f2 = self.dggm_2(f2, depth)
            # f3 = self.dggm_3(f3, depth)

            return [f0, f1, f2]