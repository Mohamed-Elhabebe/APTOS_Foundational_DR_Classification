import torch
import torch.nn as nn
import clip

class CLIPClassifier(nn.Module):
    def __init__(self, backbone_name, device):
        super(CLIPClassifier, self).__init__()
        self.device = device
        self.clip_model, _ = clip.load(backbone_name, device = self.device, jit = False)
        self.fc = nn.Linear(self.clip_model.visual.output_dim, 1)
    
    def forward(self, images):
        visual_features = self.clip_model.encode_image(images)
        visual_features = visual_features.to(torch.float32)
        logits = self.fc(visual_features)
        return logits