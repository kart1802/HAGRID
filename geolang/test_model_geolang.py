import torch
import torch.nn as nn
import torch.nn.functional as F


from types import SimpleNamespace
import sys
import os

# Add project root and mamba_clip to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mamba_clip")))


from model.geolang import geolang


# --------------------------------------------------
# TEST SCRIPT
# --------------------------------------------------
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = SimpleNamespace(
        use_dggm=True,
        use_adci=True,
        use_pretrained_mamba_clip=True,
        use_pretrained_clip=True,
        dggm_max_tokens=4096,
        adci_align_channels=256,
        adci_out_channels=128,
        adci_groups=1,
        mamba_clip_pretrain="/home/tejass/Downloads/TUDELFT_ROBOTICS/Robotics_Q3/CV/VMamba_B_clip.pt",
        clip_pretrain="/home/tejass/Downloads/TUDELFT_ROBOTICS/Robotics_Q3/CV/HAGRID/geolang/pretrain/RN50.pt",
        word_len=20
    )

    model = geolang(cfg).to(device).eval()

    B, H, W = 1, 416,416
    L = 20

    img = torch.randn(B, 3, H, W).to(device)
    depth = torch.randn(B, 1, H, W).to(device)
    word = torch.randint(0, 1000, (B, L)).to(device)

    with torch.no_grad():
        out = model(img, word, depth)

    print("\nForward pass OK\n")

    for i, v in enumerate(out["vis"]):
        print(f"vis[{i}] shape:", tuple(v.shape))

    print("adci:", None if out["adci"] is None else tuple(out["adci"].shape))
    print("global:", tuple(out["global"].shape))
    print("word:", tuple(out["word"].shape)) # word: (1, 20, 512)
#state: (1, 1024)
    print("state:", tuple(out["state"].shape))
    print("pad_mask:", tuple(out["pad_mask"].shape))