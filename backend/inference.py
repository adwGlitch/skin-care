import sys
import os
from PIL import Image

# Add the parent directory to sys.path so we can import the ai module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ai.src.predict import DermaAIPredictor
except ImportError:
    print("Failed to import ai module. Ensure the 'ai' directory is accessible.")
    raise

class InferenceService:
    def __init__(self):
        # The predictor automatically looks for best_model.pt
        self.predictor = DermaAIPredictor()
        
    def is_model_loaded(self) -> bool:
        return self.predictor.model_loaded

    def analyze(self, image: Image.Image) -> dict:
        """
        Runs the full inference pipeline (quality check + prediction + Grad-CAM).
        Returns a dictionary matching the AnalysisResponse schema structure.
        """
        result = self.predictor.predict(image)
        
        # If rejected by quality check, return early
        if result.get("status") == "error":
            return result
            
        # The frontend expects top_predictions to have a 'class' key, but python models can't use 'class' as a kwarg easily if mapped directly. 
        # The schemas.py maps it to 'class_name', but the json response we return can map it properly.
        # Ensure the dictionary returned matches the JSON structure needed by the frontend.
        formatted_top_preds = []
        for p in result["top_predictions"]:
            formatted_top_preds.append({
                "class_name": p["class"],
                "name": p["name"],
                "probability": p["probability"]
            })
            
        result["top_predictions"] = formatted_top_preds
        return result

# Singleton instance
inference_service = InferenceService()
