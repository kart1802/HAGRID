"""
Run Mamba-CLIP inference on a single image against a list of text prompts.

Usage:
    python infer_single_image.py

Edit the CONFIGURATION section below to change image path, text prompts,
model, checkpoint, or device.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image

import sys
from pathlib import Path

# Setup path to allow imports and set package context
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

# Add parent to path so mamba_clip can be imported as a package
sys.path.insert(0, str(parent_dir))

# Import from the package
from mamba_clip import models
from mamba_clip.tokenizer import SimpleTokenizer
import os

project_root = script_dir.parent
repo_root = project_root.parent
df_root = repo_root.parent

VMAMBA_DIR = df_root / "VMamba"
PRETRAIN_DIR = project_root / "pretrain"
test_path_str = str(VMAMBA_DIR)
model_path_str = str(PRETRAIN_DIR)



## ============ CONFIGURATION — edit these directly ============ ##
IMAGE_PATH = test_path_str + "/assets/architecture.png"
TEXTS = [ "a photo of a dog", "a photo of a cat", "a photo of a toothpaste"]
MODEL_NAME = "CLIP_VMamba_B"
CHECKPOINT = "/home/tejass/Downloads/TUDELFT_ROBOTICS/Robotics_Q3/CV/HAGRID/geolang/pretrain/VMamba_B_clip.pt"          # path to .pt checkpoint, or "" for random weights
GPU = 0                  # set to -1 for CPU
## ============================================================= ##


@torch.no_grad()
def main():
    device = torch.device(f"cuda:{GPU}" if GPU >= 0 and torch.cuda.is_available() else "cpu")

    # ---- build model --------------------------------------------------------
    print(f"=> Creating model: {MODEL_NAME}")
    model = getattr(models, MODEL_NAME)(mask_ratio=0.0)
    
    layer_outpts = {}
    
    def save_output(name):
        def hook(module, input, output):
            layer_outpts[name] = output
        return hook
    for i,layer in enumerate(model.visual.layers):
        layer.register_forward_hook(save_output(f"layer_{i}"))
        

    if CHECKPOINT:
        print(f"=> Loading checkpoint: {CHECKPOINT}")
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # handle DataParallel / DDP "module." prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device).eval()

    # ---- preprocess image ---------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize(416),
        transforms.CenterCrop(416),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(IMAGE_PATH).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # 1, 3, 224, 224

    # ---- tokenize text prompts ----------------------------------------------
    tokenizer = SimpleTokenizer()
    text_tokens = tokenizer(TEXTS).to(device)  # N_texts, 77

    # ---- encode -------------------------------------------------------------
    image_features, embeddings = model.encode_image(img_tensor) 
    print (embeddings[0].shape,"embeddings")  # 1, embed_dim
    print ("Image features shape:", image_features.shape)
    text_features  = model.encode_text(text_tokens)                 # N, embed_dim
    print ("Text features shape:", text_features.shape)
    
    for k,v in layer_outpts.items():
        print (f"{k}: {v.shape}")

    # L2-normalise
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity → probabilities
    logits = (image_features @ text_features.T).squeeze(0)          # N
    probs  = logits.softmax(dim=-1)

    # ---- print results ------------------------------------------------------
    print("\n=== Results ===")
    for txt, p in sorted(zip(TEXTS, probs.tolist()), key=lambda x: -x[1]):
        print(f"  {p*100:5.1f}%  {txt}")


if __name__ == "__main__":
    main()
