from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.predictor import ShelfVisionPredictor
from app.schemas import PredictionResponse

app = FastAPI(
    title="Retail Shelf Vision Service",
    description="Production-oriented computer vision inference API for retail shelf condition classification",
    version="1.0.0",
)

predictor = None


@app.on_event("startup")
def startup_event():
    global predictor
    predictor = ShelfVisionPredictor()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

    try:
        content = await file.read()
        image = Image.open(BytesIO(content))
        result = predictor.predict(image)
        return PredictionResponse(**result)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc