import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from model.DGGM import DGGM
from model.ADCI import ADCI

from types import SimpleNamespace
import mamba_clip.models as mamba_models

print ("Importing geolang...")

class geolang(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # ---------------- FLAGS ----------------
        self.use_dggm = cfg.use_dggm
        self.use_adci = cfg.use_adci
        self.use_pretrained_mamba_clip = cfg.use_pretrained_mamba_clip
        self.use_pretrained_clip = cfg.use_pretrained_clip

        self.dggm_max_tokens = cfg.dggm_max_tokens

        # ---------------- BACKBONE ----------------
        print(f"Load pretrained Mamba-CLIP: {self.use_pretrained_mamba_clip}")
        self.backbone = mamba_models.CLIP_VMamba_B(mask_ratio=0.0)

        if self.use_pretrained_mamba_clip:
            ckpt = torch.load(cfg.mamba_clip_pretrain, map_location="cpu", weights_only=False)            
            state_dict = ckpt.get("state_dict", ckpt)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        # ---------------- TEXT ENCODER ----------------
        print(f"Load pretrained CLIP: {self.use_pretrained_clip}")
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.backbone_text = build_model(
            clip_model.state_dict(),
            cfg.word_len,
            self.use_pretrained_clip
        ).float()

        # ---------------- DGGM ----------------
        if self.use_dggm:
            print("Initializing DGGM")

            self.dggm_blocks = nn.ModuleList([
                DGGM(256),
                DGGM(512),
                DGGM(1024),
            ])
        else:
            self.dggm_blocks = None

        # ---------------- ADCI ----------------
        if self.use_adci:
            print("Initializing ADCI")

            self.adci = ADCI(
                in_channels_list=[256, 512, 1024],
                align_channels=cfg.adci_align_channels,
                out_channels=cfg.adci_out_channels,
                L=3,
                G=cfg.adci_groups,
                target_size=(52, 52),
            )
        else:
            self.adci = None

    # --------------------------------------------------
    # Utility: ensure tensor is BCHW
    # --------------------------------------------------
    def _to_bchw(self, x):
        if x.shape[1] > x.shape[-1]:
            return x
        else:
            return x.permute(0, 3, 1, 2).contiguous() # B C H W

    # --------------------------------------------------
    # DGGM Application
    # --------------------------------------------------
    def _apply_dggm(self, vis, depth):
        if not self.use_dggm:
            return [self._to_bchw(v) for v in vis] 

        out = []

        for i, v in enumerate(vis):
            v = self._to_bchw(v) 
            B, C, H, W = v.shape

            # Skip highest resolution (expensive)
            # if i == 0 or H * W > self.dggm_max_tokens:
            #     out.append(v)
            #     continue

            v = v.permute(0, 2, 3, 1)  # B H W C
            v = self.dggm_blocks[i](v, depth)
            v = v.permute(0, 3, 1, 2).contiguous()

            out.append(v)
            print (f"Applied DGGM to vis[{i}] with shape {tuple(v.shape)}")

        return out

    # --------------------------------------------------
    # ADCI Application
    # --------------------------------------------------
    def _apply_adci(self, vis):
        if not self.use_adci:
            return None

        feats = [self._to_bchw(v) for v in vis[:3]] # Only use C0, C1, C2 for ADCI as per paper
        print ("ADCI Used")
        return self.adci(feats)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, img, word, depth=None):

        pad_mask = (word == 0)

        # -------- Vision --------
        global_feat, vis = self.backbone.encode_image(img)

        vis = self._apply_dggm(vis, depth)
        
        adci_out = self._apply_adci(vis)

        # -------- Text --------
        word_feat, state = self.backbone_text.encode_text(word)

        return {
            "vis": vis,
            "adci": adci_out,
            "global": global_feat,
            "word": word_feat,
            "state": state,
            "pad_mask": pad_mask
        }

# --------------------------------------------------
# TEST SCRIPT
# --------------------------------------------------
# if __name__ == "__main__":

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     cfg = SimpleNamespace(
#         use_dggm=True,
#         use_adci=True,
#         use_pretrained_mamba_clip=True,
#         use_pretrained_clip=True,
#         dggm_max_tokens=4096,
#         adci_align_channels=256,
#         adci_out_channels=512,
#         adci_groups=1,
#         mamba_clip_pretrain="/home/tejass/Downloads/TUDELFT_ROBOTICS/Robotics_Q3/CV/VMamba_B_clip.pt",
#         clip_pretrain="/home/tejass/Downloads/TUDELFT_ROBOTICS/Robotics_Q3/CV/HAGRID/geolang/pretrain/RN50.pt",
#         word_len=20
#     )

#     model = geolang(cfg).to(device).eval()

#     B, H, W = 1, 224, 224
#     L = 20

#     img = torch.randn(B, 3, H, W).to(device)
#     depth = torch.randn(B, 1, H, W).to(device)
#     word = torch.randint(0, 1000, (B, L)).to(device)

#     with torch.no_grad():
#         out = model(img, word, depth)

#     print("\nForward pass OK\n")

#     for i, v in enumerate(out["vis"]):
#         print(f"vis[{i}] shape:", tuple(v.shape))

#     print("adci:", None if out["adci"] is None else tuple(out["adci"].shape))
#     print("global:", tuple(out["global"].shape))
#     print("word:", tuple(out["word"].shape))
#     print("state:", tuple(out["state"].shape))
#     print("pad_mask:", tuple(out["pad_mask"].shape))