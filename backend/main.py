from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import uuid

try:
    from .schemas import AnalysisResponse
    from .inference import inference_service
except ImportError:
    from schemas import AnalysisResponse
    from inference import inference_service

app = FastAPI(title="DermaAI ML Backend", version="1.0")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": inference_service.is_model_loaded(),
        "message": "DermaAI ML Backend is running."
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
    try:
        # Run inference
        result_dict = inference_service.analyze(img)
        
        # Add scan ID
        scan_id = str(uuid.uuid4())[:8]
        
        if result_dict.get("status") == "error":
            return AnalysisResponse(
                status="error",
                message=result_dict.get("message"),
                image_quality=result_dict.get("image_quality"),
                scan_id=scan_id
            )
            
        top_preds = []
        for p in result_dict.get("top_predictions", []):
            # p comes from inference_service as {'class_name': ..., 'name': ..., 'probability': ...}
            # The AnalysisResponse schema expects 'class' in json or kwargs by alias.
            # We can instantiate the Pydantic models directly to be safe
            top_preds.append({
                "class": p["class_name"],
                "name": p["name"],
                "probability": p["probability"]
            })
            
        frontend_response = {
            "status": result_dict["status"],
            "prediction": result_dict.get("prediction"),
            "confidence": result_dict.get("confidence"),
            "image_quality": result_dict.get("image_quality"),
            "uncertainty": result_dict.get("uncertainty"),
            "heatmap_url": result_dict.get("heatmap"), 
            "model_version": result_dict.get("model_version"),
            "recommendation_level": result_dict.get("recommendation_level"),
            "scan_id": scan_id,
            "top_predictions": top_preds
        }
            
        return frontend_response
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
