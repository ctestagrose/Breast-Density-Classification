from torchvision import models
from pytorch_pretrained_vit import ViT
import torch.nn as nn

class ModelDefinition():
    def __init__(self, num_class: int, pretrained_flag=True, dropout_ratio=0.5, fc_nodes=1024, patch_size=32, img_size=int):
        self.num_class = num_class
        self.pretrain_flag = pretrained_flag
        self.dropout_ratio = dropout_ratio
        self.fc_nodes = fc_nodes
        self.patch_size=patch_size
        self.img_size=img_size

    def inception_v3(self, device='cpu'):
        model = models.inception_v3(pretrained=self.pretrain_flag)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_class))
        model.aux_logits = False
        model.to(device)
        return model

    def ViT_Pretrained(self, device='cpu'):
        if self.patch_size == 32:
            model = ViT('B_32', pretrained=True, num_classes=self.num_class, image_size=self.img_size)
        if self.patch_size == 16:
            model = ViT('B_16', pretrained=True, num_classes=self.num_class, image_size=self.img_size)
        model.aux_logits = False
        model.to(device)
        return model
