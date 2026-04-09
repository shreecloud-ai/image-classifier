from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import time
import os
from src.predict import ImageClassifier

app = FastAPI(
    title="CIFAR-10 Image Classifier API",
    description="ResNet-18 model with 95.46% accuracy",
    version="1.0"
)

# Load model once at startup (very important for performance)
classifier = ImageClassifier()

@app.get("/")
async def home():
    return {
        "message": "CIFAR-10 Image Classifier API is running 🚀",
        "model": "ResNet-18",
        "accuracy": "95.46%",
        "docs": "/docs"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    start_time = time.time()

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, etc.)")

    temp_path = None
    try:
        # Save uploaded file
        temp_path = f"temp_{file.filename}"
        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        # Get prediction
        result = classifier.predict(temp_path)

        latency = (time.time() - start_time) * 1000

        return {
            "class": result["class"],
            "confidence": result["confidence"],
            "latency_ms": round(latency, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
