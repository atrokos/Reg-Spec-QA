import argparse
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from models.gemma_model import GemmaModel
from models.phi4_model import Phi4Model
from models.aya_model import AyaModel

app = FastAPI()

model_registry = None  # Only one model can be registered at a time

class PredictionRequest(BaseModel):
    image: str
    prompt: str

class ModelRegistration(BaseModel):
    model_name: str

# FastAPI endpoint for prediction
@app.post("/predict")
async def predict(request: PredictionRequest):
    if model_registry is None:
        raise HTTPException(status_code=400, detail="No model registered")

    model = model_registry

    # Generate prediction
    try:
        if isinstance(model, GemmaModel):
            result = model.generate((request.image, request.prompt))
        elif isinstance(model, Phi4Model):
            result = model.generate((request.image, request.prompt))
        elif isinstance(model, AyaModel):
            result = model.generate((request.image, request.prompt))
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"result": result}

# FastAPI endpoint for registering a model
@app.post("/register")
async def register_model(request: ModelRegistration):
    global model_registry

    if request.model_name == "gemma":
        model_registry = GemmaModel()
    elif request.model_name == "phi4":
        model_registry = Phi4Model()
    elif request.model_name == "aya":
        model_registry = AyaModel()
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")

    return {"status": "model registered"}

def run_offline(input_path: str, output_path: str):
    # TODO: Implement offline mode
    raise NotImplementedError("Offline mode is not implemented")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the application in FastAPI or offline mode.")
    parser.add_argument("--mode", choices=["fastapi", "offline"], required=True, help="Mode to run the application")
    parser.add_argument("--input", type=str, help="Input JSON file path for offline mode")
    parser.add_argument("--output", type=str, help="Output JSON file path for offline mode")

    args = parser.parse_args()

    if args.mode == "fastapi":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.mode == "offline":
        if not args.input or not args.output:
            print("Input and output paths are required in offline mode.")
        else:
            run_offline(args.input, args.output)
