import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights

try:
    from .attention import CBAMBlock
except ImportError:
    from attention import CBAMBlock

class DermaAI_MobileNetV3(nn.Module):
    def __init__(self, num_classes=7, use_attention=True, pretrained=True):
        super(DermaAI_MobileNetV3, self).__init__()
        
        self.use_attention = use_attention
        
        # Load backbone
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)
        
        # Extract features (up to the average pooling layer)
        self.features = mobilenet.features
        
        # MobileNetV3 Large feature dimension before classifier is 960
        feature_dim = 960 
        
        # Optional Attention Block
        if self.use_attention:
            self.attention = CBAMBlock(in_planes=feature_dim)
            
        # Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head (replaces the original MobileNet classifier)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # Backbone features
        x = self.features(x)
        
        # Attention Refinement
        if self.use_attention:
            x = self.attention(x)
            
        # Pooling
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        return x

    def extract_features_and_attention(self, x):
        """
        Helper method for Grad-CAM.
        Returns the feature map right before pooling, and the final logits.
        """
        features = self.features(x)
        if self.use_attention:
            features = self.attention(features)
        
        pooled = self.pool(features)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        
        return features, logits
