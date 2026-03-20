import torch
import torch.nn as nn

from model.DGGM import DGGM
from model.ADCI import ADCI

import mamba_clip.models as mamba_models

print ("Importing geolang...")

class geolang(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # ---------------- FLAGS ----------------
        self.use_dggm = getattr(cfg, "use_dggm", False)
        self.use_adci = getattr(cfg, "use_adci", False)
        self.use_pretrained_mamba_clip = getattr(cfg, "use_pretrained_mamba_clip", True)
        self.use_bert_text = getattr(cfg, "use_bert_text", True)

        self.word_len = getattr(cfg, "word_len", 20)
        self.word_dim = getattr(cfg, "word_dim", 1024)
        self.bert_model_name = getattr(cfg, "bert_model_name", "bert-base-uncased")

        self.dggm_max_tokens = getattr(cfg, "dggm_max_tokens", 4096)

        # ---------------- BACKBONE ----------------
        print(f"Load pretrained Mamba-CLIP: {self.use_pretrained_mamba_clip}")
        self.backbone = mamba_models.CLIP_VMamba_B(mask_ratio=0.0)

        if self.use_pretrained_mamba_clip:
            ckpt = torch.load(cfg.mamba_clip_pretrain, map_location="cpu", weights_only=False)            
            state_dict = ckpt.get("state_dict", ckpt)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        # ---------------- TEXT ENCODER (BERT) ----------------
        if self.use_bert_text:
            print(f"Load pretrained BERT: {self.bert_model_name}")
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "BERT text encoder requires `transformers`. "
                    "Install with: pip install transformers"
                ) from exc

            self.text_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.text_encoder = AutoModel.from_pretrained(self.bert_model_name)
            hidden_size = self.text_encoder.config.hidden_size
            self.text_proj = nn.Identity() if hidden_size == self.word_dim else nn.Linear(hidden_size, self.word_dim)
        else:
            self.text_tokenizer = None
            self.text_encoder = None
            self.text_proj = nn.Identity()

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
                align_channels=getattr(cfg, "adci_align_channels", 256),
                out_channels=getattr(cfg, "adci_out_channels", 512),
                L=3,
                G=getattr(cfg, "adci_groups", 1),
                target_size=(26, 26),
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

            if H * W > self.dggm_max_tokens:
                out.append(v)
                continue

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
    def _encode_text(self, word=None, sentence=None, device=None):
        if not self.use_bert_text:
            word_feat, state = self.backbone.encode_text(word)
            pad_mask = (word == 0)
            return word_feat, state, pad_mask

        if sentence is not None:
            encoded = self.text_tokenizer(
                sentence,
                padding='max_length',
                truncation=True,
                max_length=self.word_len,
                return_tensors='pt'
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
        else:
            input_ids = word.long().to(device)
            attention_mask = (input_ids != 0).long()
            if input_ids.max().item() >= self.text_encoder.config.vocab_size:
                raise ValueError(
                    "Input token ids exceed BERT vocab size. "
                    "Pass raw text via `sentence` so BERT tokenizer can be used."
                )

        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        word_feat = self.text_proj(text_out.last_hidden_state)

        if text_out.pooler_output is None:
            state = word_feat[:, 0]
        else:
            state = self.text_proj(text_out.pooler_output)

        pad_mask = ~attention_mask.bool()
        return word_feat, state, pad_mask

    def forward(self, img, word, depth=None, mask=None, grasp_qua_mask=None, grasp_sin_mask=None, grasp_cos_mask=None, grasp_wid_mask=None, sentence=None):

        # -------- Vision --------
        global_feat, vis = self.backbone.encode_image(img)

        vis = self._apply_dggm(vis, depth)
        
        adci_out = self._apply_adci(vis)

        # -------- Text --------
        word_feat, state, pad_mask = self._encode_text(word=word, sentence=sentence, device=img.device)

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
#         use_bert_text=True,
#         bert_model_name="bert-base-uncased",
#         dggm_max_tokens=4096,
#         adci_align_channels=256,
#         adci_out_channels=512,
#         adci_groups=1,
#         mamba_clip_pretrain="/home/tejass/Downloads/TUDELFT_ROBOTICS/Robotics_Q3/CV/VMamba_B_clip.pt",
#         word_dim=1024,
#         word_len=20
#     )

#     model = geolang(cfg).to(device).eval()

#     B, H, W = 1, 224, 224
#     L = 20

#     img = torch.randn(B, 3, H, W).to(device)
#     depth = torch.randn(B, 1, H, W).to(device)
#     word = torch.randint(0, 30000, (B, L)).to(device)
#     sentence = ["pick up the red mug"]

#     with torch.no_grad():
#         out = model(img, word, depth, sentence=sentence)

#     print("\nForward pass OK\n")

#     for i, v in enumerate(out["vis"]):
#         print(f"vis[{i}] shape:", tuple(v.shape))

#     print("adci:", None if out["adci"] is None else tuple(out["adci"].shape))
#     print("global:", tuple(out["global"].shape))
#     print("word:", tuple(out["word"].shape))
#     print("state:", tuple(out["state"].shape))
#     print("pad_mask:", tuple(out["pad_mask"].shape))