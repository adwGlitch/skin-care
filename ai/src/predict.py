import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import base64
from io import BytesIO

try:
    from .model import DermaAI_MobileNetV3
    from .preprocessing import get_transforms, REVERSE_CLASS_MAPPING, CLASS_MAPPING
    from .explainability import GradCAM, overlay_heatmap
except ImportError:
    from model import DermaAI_MobileNetV3
    from preprocessing import get_transforms, REVERSE_CLASS_MAPPING, CLASS_MAPPING
    from explainability import GradCAM, overlay_heatmap

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Human readable names for the frontend
FULL_CLASS_NAMES = {
    'akiec': 'Actinic keratoses / intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevus',
    'vasc': 'Vascular lesion'
}

class DermaAIPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DermaAI_MobileNetV3(num_classes=7, use_attention=True, pretrained=False)
        
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "best_model.pt")
            
        self.model_loaded = False
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model_loaded = True
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: No model found at {model_path}. Inference will run with random weights (for testing).")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transforms = get_transforms(is_train=False)
        self.grad_cam = GradCAM(self.model)

    def check_image_quality(self, image: Image.Image):
        """
        Basic image quality gate.
        Returns: score (0-1), status ('good', 'poor', 'rejected'), messages
        """
        messages = []
        status = "good"
        score = 1.0
        
        # 1. Resolution Check
        width, height = image.size
        if width < 224 or height < 224:
            status = "rejected"
            score = 0.0
            messages.append(f"Image resolution too low ({width}x{height}). Minimum is 224x224.")
            return score, status, messages
            
        # 2. Blur Check (Variance of Laplacian)
        # Convert to cv2 grayscale
        import cv2
        img_cv = np.array(image.convert('L'))
        laplacian_var = cv2.Laplacian(img_cv, cv2.CV_64F).var()
        
        if laplacian_var < 50:
            status = "poor"
            score *= 0.5
            messages.append("Image appears extremely blurry. Ensure the lesion is in focus.")
            
        # 3. Lighting Check
        mean_brightness = np.mean(img_cv)
        if mean_brightness < 40:
            status = "poor"
            score *= 0.6
            messages.append("Image is too dark.")
        elif mean_brightness > 240:
            status = "poor"
            score *= 0.6
            messages.append("Image is overexposed/too bright.")
            
        return max(0.1, score), status, messages

    def _image_to_base64(self, image: Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict(self, image: Image.Image):
        """
        Runs inference and generates explainability overlay.
        """
        # Quality Check
        q_score, q_status, q_msg = self.check_image_quality(image)
        if q_status == "rejected":
            return {
                "status": "error",
                "message": "Image rejected by quality gate.",
                "image_quality": {"score": q_score, "status": q_status, "messages": q_msg}
            }

        # Preprocess
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        # Get top 3 predictions
        top_indices = np.argsort(probs)[::-1][:3]
        
        top_predictions = []
        for idx in top_indices:
            class_abbr = REVERSE_CLASS_MAPPING[idx]
            top_predictions.append({
                "class": class_abbr,
                "name": FULL_CLASS_NAMES[class_abbr],
                "probability": float(probs[idx])
            })
            
        top_class = top_predictions[0]["class"]
        confidence = top_predictions[0]["probability"]
        
        # Uncertainty heuristic
        uncertainty = "low"
        if confidence < 0.5:
            uncertainty = "high"
        elif confidence < 0.75:
            uncertainty = "moderate"
            
        # Generate Grad-CAM Heatmap for the top class
        # (Requires gradients, so we use the GradCAM class which manages this)
        # Note: We pass the tensor again through the model with gradients enabled inside generate_heatmap
        cam, _ = self.grad_cam.generate_heatmap(input_tensor, target_class=top_indices[0])
        heatmap_img = overlay_heatmap(image, cam)
        heatmap_b64 = self._image_to_base64(heatmap_img)

        return {
            "status": "success",
            "prediction": top_class,
            "confidence": float(confidence),
            "top_predictions": top_predictions,
            "image_quality": {
                "score": float(q_score),
                "status": q_status,
                "messages": q_msg
            },
            "uncertainty": uncertainty,
            "heatmap": heatmap_b64,
            "model_version": "v1.0-mobilenetv3-attention",
            "recommendation_level": "professional_evaluation"
        }
