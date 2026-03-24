import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from model.DGGM import DGGM
from model.ADCI import ADCI
from .layers import FPN, Projector, TransformerDecoder, MultiTaskProjector

from types import SimpleNamespace
import mamba_clip.models as mamba_models
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

PRETRAIN_DIR = project_root / "pretrain"
model_path_str = str(PRETRAIN_DIR)



class geolang(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # ---------------- FLAGS ----------------
        self.use_dggm = cfg.use_dggm
        self.use_adci = cfg.use_adci
        self.use_pretrained_mamba_clip = cfg.use_pretrained_mamba_clip
        self.use_pretrained_clip = cfg.use_pretrained_clip
        self.use_contrastive = getattr(cfg, "use_contrastive", True)
        self.use_grasp_masks = getattr(cfg, "use_grasp_masks", True)

        # one-time debug flag to print ADCI input/output shapes
        self._adci_debug_done = False

        self.dggm_max_tokens = cfg.dggm_max_tokens
        self.word_dim = getattr(cfg, "word_dim", 1024)
        self.vis_dim = getattr(cfg, "vis_dim", 512)
        self.fpn_in = getattr(cfg, "fpn_in", [128])
        self.fpn_out = getattr(cfg, "fpn_out", [128])
        self.fpn_vis_dim = self.fpn_out[0]
        self.DGGM_input_channels = [512, 1024, 1024]

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
                DGGM(self.DGGM_input_channels[0]),
                DGGM(self.DGGM_input_channels[1]),
                DGGM(self.DGGM_input_channels[2]),
            ])
        else:
            self.dggm_blocks = None

        # ---------------- ADCI ----------------
        if self.use_adci:
            print("Initializing ADCI")

            self.adci = ADCI(
                in_channels_list=[512, 1024, 1024],  # C2, C3, C4 channels from the backbone
                align_channels=cfg.adci_align_channels,
                out_channels=cfg.adci_out_channels,
                L=3,
                G=cfg.adci_groups,
                target_size=(26, 26),
            )
        else:
            self.adci = None

        # Build pseudo multi-scale features from a single ADCI embedding, then reuse the
        # existing FPN/decoder/projector stack.
        # self.adci_to_v3 = nn.Sequential(
        #     nn.Conv2d(cfg.adci_out_channels, self.fpn_in[0], kernel_size=1, bias=False),
        #     nn.BatchNorm2d(self.fpn_in[0]),
        #     nn.ReLU(True),
        # )
        self.adci_to_v4 = nn.Sequential(
            nn.Conv2d(cfg.adci_out_channels, self.fpn_in[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(self.fpn_in[0]),
            nn.ReLU(True),
        )
        # self.adci_to_v5 = nn.Sequential(
        #     nn.Conv2d(cfg.adci_out_channels, self.fpn_in[2], kernel_size=1, bias=False),
        #     nn.BatchNorm2d(self.fpn_in[2]),
        #     nn.ReLU(True),
        # )

        self.neck = FPN(
            in_channels=self.fpn_in,
            out_channels=self.fpn_out,
            txt_dim=self.word_dim,
        )

        # Align single-embedding FPN channels to decoder/text dimension.
        if self.fpn_vis_dim != self.vis_dim:
            self.fpn_to_decoder = nn.Sequential(
                nn.Conv2d(self.fpn_vis_dim, self.vis_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.vis_dim),
                nn.ReLU(True),
            )
        else:
            self.fpn_to_decoder = nn.Identity()

        if self.use_contrastive:
            print("Use contrastive learning module")
            self.decoder = TransformerDecoder(
                num_layers=getattr(cfg, "num_layers", 3),
                d_model=self.vis_dim,
                nhead=getattr(cfg, "num_head", 8),
                dim_ffn=getattr(cfg, "dim_ffn", 2048),
                dropout=getattr(cfg, "dropout", 0.1),
                return_intermediate=getattr(cfg, "intermediate", False),
            )
        else:
            print("Disable contrastive learning module")
            self.decoder = None

        if self.use_grasp_masks:
            print("Use grasp masks")
            self.proj = MultiTaskProjector(self.word_dim, self.vis_dim // 2, 3)
        else:
            print("Disable grasp masks")
            self.proj = Projector(self.word_dim, self.vis_dim // 2, 3)

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
        if (not self.use_dggm) or (depth is None):
            return [self._to_bchw(v) for v in vis] 

        out = []

        for i, v in enumerate(vis):
            v = self._to_bchw(v)
            v = v.permute(0, 2, 3, 1)
            with torch.amp.autocast("cuda", enabled=False):
                v = self.dggm_blocks[i](v.float(), depth.float())
            v = v.permute(0, 3, 1, 2).contiguous()

            out.append(v)
            # print (f"Applied DGGM to vis[{i}] with shape {tuple(v.shape)}")

        return out

    # --------------------------------------------------
    # ADCI Application
    # --------------------------------------------------
    def _apply_adci(self, vis):
        if not self.use_adci:
            return None

        # Only use C0, C1, C2 for ADCI as per paper
        # When DGGM is disabled, these come directly from the backbone.
        if not self._adci_debug_done:
            try:
                print(f"[ADCI debug] num vis features: {len(vis)}")
                for i, v in enumerate(vis):
                    if hasattr(v, "shape"):
                        print(f"[ADCI debug] vis[{i}] shape before _to_bchw: {tuple(v.shape)}")
            except Exception:
                pass

        feats = [self._to_bchw(v) for v in vis[:3]]

        if not self._adci_debug_done:
            try:
                for i, f in enumerate(feats):
                    print(f"[ADCI debug] feats[{i}] shape for ADCI: {tuple(f.shape)}")
            except Exception:
                pass

        adci_out = self.adci(feats)

        if not self._adci_debug_done:
            try:
                print(f"[ADCI debug] ADCI output shape: {tuple(adci_out.shape)}")
            except Exception:
                pass
            self._adci_debug_done = True

        return adci_out

    def _single_embedding_to_fpn_inputs(self, adci_feat):
        # ADCI output is the single source embedding (26x26). Derive v3/v4/v5 from it.
        v4 = adci_feat
        # v3 = F.interpolate(v4, scale_factor=2, mode="bilinear", align_corners=False)
        # v5 = F.avg_pool2d(v4, kernel_size=2, stride=2)

        # v3 = self.adci_to_v3(v3)
        v4 = self.adci_to_v4(v4)
        # v5 = self.adci_to_v5(v5)
        return [v4]
    


    def _parse_forward_inputs(self, args, kwargs):
        depth = kwargs.get("depth", None)
        mask = kwargs.get("mask", None)
        grasp_qua_mask = kwargs.get("grasp_qua_mask", None)
        grasp_sin_mask = kwargs.get("grasp_sin_mask", None)
        grasp_cos_mask = kwargs.get("grasp_cos_mask", None)
        grasp_wid_mask = kwargs.get("grasp_wid_mask", None)

        # Backward compatible paths:
        # 1 extra positional arg -> depth (test script style).
        # 5+ extra positional args -> segmentation/grasp targets (CROG engine style).
        if len(args) == 1 and depth is None and mask is None:
            depth = args[0]
        elif len(args) >= 5:
            mask = args[0]
            grasp_qua_mask = args[1]
            grasp_sin_mask = args[2]
            grasp_cos_mask = args[3]
            grasp_wid_mask = args[4]
            if len(args) >= 6 and depth is None:
                depth = args[5]

        return depth, mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, img, word, *args, **kwargs):

        depth, mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask = self._parse_forward_inputs(args, kwargs)
        
        

        pad_mask = (word == 0)

        # -------- Vision --------
        global_feat, vis = self.backbone.encode_image(img)
  

        vis = self._apply_dggm(vis, depth)
        
        adci_out = self._apply_adci(vis)

        # -------- Text --------
        # print (f"word.shape: {tuple(word.shape)}")
        word_feat, state = self.backbone_text.encode_text(word)
        # print (f"word_feat shape:", tuple(word_feat.shape))
        # print (f"state shape:", tuple(state.shape))

        pred = None
        grasp_qua_pred = None
        grasp_sin_pred = None
        grasp_cos_pred = None
        grasp_wid_pred = None

        if adci_out is not None:
            fpn_inputs = self._single_embedding_to_fpn_inputs(adci_out)
            fq = self.neck(fpn_inputs, state)
            fq = self.fpn_to_decoder(fq)
            # print (f"FPN output shape:", tuple(fq.shape)) # matched with CROG
            b, c, h, w = fq.size()

            if self.use_contrastive and self.decoder is not None:
                fq = self.decoder(fq, word_feat, pad_mask)
                # print (f"Decoder output shape:", tuple(fq.shape))
                if isinstance(fq, list):
                    fq = fq[-1]
                fq = fq.reshape(b, c, h, w)

            if self.use_grasp_masks:
                pred, grasp_qua_pred, grasp_sin_pred, grasp_cos_pred, grasp_wid_pred = self.proj(fq, state)
            else:
                pred = self.proj(fq, state)

        # CROG-compatible training/eval outputs when masks are provided.
        if mask is not None:
            if self.use_grasp_masks:
                if self.training:
                    if pred.shape[-2:] != mask.shape[-2:]:
                        mask = F.interpolate(mask, pred.shape[-2:], mode="nearest").detach()
                        grasp_qua_mask = F.interpolate(grasp_qua_mask, grasp_qua_pred.shape[-2:], mode="nearest").detach()
                        grasp_sin_mask = F.interpolate(grasp_sin_mask, grasp_sin_pred.shape[-2:], mode="nearest").detach()
                        grasp_cos_mask = F.interpolate(grasp_cos_mask, grasp_cos_pred.shape[-2:], mode="nearest").detach()
                        grasp_wid_mask = F.interpolate(grasp_wid_mask, grasp_wid_pred.shape[-2:], mode="nearest").detach()

                    weight = mask * 0.5 + 1
                    loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
                    grasp_qua_loss = F.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
                    grasp_sin_loss = F.smooth_l1_loss(grasp_sin_pred, grasp_sin_mask)
                    grasp_cos_loss = F.smooth_l1_loss(grasp_cos_pred, grasp_cos_mask)
                    grasp_wid_loss = F.smooth_l1_loss(grasp_wid_pred, grasp_wid_mask)

                    total_loss = loss + grasp_qua_loss + grasp_sin_loss + grasp_cos_loss + grasp_wid_loss
                    
                    # Check for NaN values in losses
                    if torch.isnan(loss) or torch.isnan(grasp_qua_loss) or torch.isnan(grasp_sin_loss) or torch.isnan(grasp_cos_loss) or torch.isnan(grasp_wid_loss):
                        import sys
                        print("\n⚠️ WARNING: NaN detected in loss computation!", file=sys.stderr)
                        print(f"   loss: {loss.item()}", file=sys.stderr)
                        print(f"   grasp_qua_loss: {grasp_qua_loss.item()}", file=sys.stderr)
                        print(f"   grasp_sin_loss: {grasp_sin_loss.item()}", file=sys.stderr)
                        print(f"   grasp_cos_loss: {grasp_cos_loss.item()}", file=sys.stderr)
                        print(f"   grasp_wid_loss: {grasp_wid_loss.item()}", file=sys.stderr)
                        print(f"   Pred stats - min: {pred[~torch.isnan(pred)].min().item() if (~torch.isnan(pred)).any() else 'all NaN'}, max: {pred[~torch.isnan(pred)].max().item() if (~torch.isnan(pred)).any() else 'all NaN'}", file=sys.stderr)
                        print(f"   Grasp QUA pred stats - min: {grasp_qua_pred[~torch.isnan(grasp_qua_pred)].min().item() if (~torch.isnan(grasp_qua_pred)).any() else 'all NaN'}, max: {grasp_qua_pred[~torch.isnan(grasp_qua_pred)].max().item() if (~torch.isnan(grasp_qua_pred)).any() else 'all NaN'}", file=sys.stderr)
                    
                    loss_dict = {
                        "m_ins": loss.item(),
                        "m_qua": grasp_qua_loss.item(),
                        "m_sin": grasp_sin_loss.item(),
                        "m_cos": grasp_cos_loss.item(),
                        "m_wid": grasp_wid_loss.item(),
                    }
                    return (
                        (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()),
                        (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask),
                        total_loss,
                        loss_dict,
                    )

                return (
                    (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()),
                    (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask),
                )

            if self.training:
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:], mode="nearest").detach()

                loss = F.binary_cross_entropy_with_logits(pred, mask)
                loss_dict = {
                    "m_ins": loss.item(),
                    "m_qua": 0,
                    "m_sin": 0,
                    "m_cos": 0,
                    "m_wid": 0,
                }
                return (pred.detach(), None, None, None, None), (mask, None, None, None, None), loss, loss_dict

            return (pred.detach(), None, None, None, None), (mask, None, None, None, None)

        return {
            "vis": vis,
            "adci": adci_out,
            "fpn": None if adci_out is None else fq,
            "global": global_feat,
            "word": word_feat,
            "state": state,
            "pad_mask": pad_mask,
            "seg_mask": pred,
            "grasp_qua": grasp_qua_pred,
            "grasp_sin": grasp_sin_pred,
            "grasp_cos": grasp_cos_pred,
            "grasp_wid": grasp_wid_pred,
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
#         mamba_clip_pretrain=f"{model_path_str}/VMamba_B_clip.pt",
#         clip_pretrain=f"{model_path_str}/RN50.pt",
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