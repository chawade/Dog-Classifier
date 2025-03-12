from fastapi import FastAPI, File, UploadFile
from app.api.predict import predict_breed

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Dog Breed Classifier API"}

@app.post("/predict/")
async def classify_dog(image: UploadFile = File(...)):
    result = predict_breed(image)
    return {"breed": result}
