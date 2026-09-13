import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

class GradCAM:
    """
    Grad-CAM for DermaAI_MobileNetV3.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # We want to hook into the output of the attention block (or features if no attention)
        if hasattr(self.model, 'attention') and self.model.use_attention:
            target_layer = self.model.attention
        else:
            # For MobileNetV3 Large, features[-1] is the Conv2dNormActivation before pooling
            target_layer = self.model.features[-1]
            
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()
        
        # Calculate CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        return cam, target_class

def overlay_heatmap(original_img, cam, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlays the CAM heatmap onto the original image.
    Args:
        original_img: PIL Image or numpy array (RGB)
        cam: 2D numpy array [0, 1]
    Returns:
        PIL Image with heatmap overlay
    """
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img)
        
    # Resize CAM to match image
    cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    
    # Convert CAM to RGB heatmap
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * original_img)
    return Image.fromarray(overlay)
