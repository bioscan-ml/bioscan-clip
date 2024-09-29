import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

class ViT_And_MLP(nn.Module):
    def __init__(self, vit, mlp):
        super(ViT_And_MLP, self).__init__()
        self.vit = vit
        self.mlp = mlp
        for param in self.vit.parameters():
            param.requires_grad = False
    def forward(self, x):
        with torch.no_grad():
            feature = self.vit.forward_features(x).mean(dim=1)

        return self.mlp(feature)

class ViTWIthExtraLayer(nn.Module):
    def __init__(self, vit_model, new_linear_layer):
        super(ViTWIthExtraLayer, self).__init__()
        self.vit = vit_model
        self.new_linear_layer = new_linear_layer

    def get_feature(self, x):
        return self.vit(x)

    def forward(self, x):
        outputs = self.vit(x)
        outputs = self.new_linear_layer(outputs)
        return outputs
