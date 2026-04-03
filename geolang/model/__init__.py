from loguru import logger
from .geolang import geolang

def build_geolang(args):
    model = geolang(args)
    
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()
    for p in model.backbone_text.parameters():
        p.requires_grad = False
    model.backbone_text.eval()
    
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return model, trainable_params