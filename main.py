import os
import argparse
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.gemma_model import GemmaModel
from models.phi4_model import Phi4Model
from models.aya_model import AyaModel
from tqdm import tqdm
from utils import load_image_and_encode

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


def run_offline(model_name: str, language: str, split: str, dataset_path: str):
    model = None
    if model_name == "gemma":
        model = GemmaModel()
    elif model_name == "phi4":
        model = Phi4Model()
    elif model_name == "aya":
        model = AyaModel()
    else:
        raise ValueError("Invalid model name")

    df = pd.read_csv(
        os.path.join(dataset_path, f"dataset_{language.upper()}_{split}.csv")
    )

    # TODO: make it by batch, now it's just one by one
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image_path = row["image_url"]
            encoded_image = load_image_and_encode(image_path)
            response = model.generate((encoded_image, row["question"]))
            df.loc[index, "predicted_answer"] = response

            # Use the same approach for the English question
            response_en = model.generate((encoded_image, row["question_en"]))
            df.loc[index, "predicted_answer_en"] = response_en
            df.loc[index, "model"] = model_name
            df.loc[index, "system_prompt"] = model.system_prompt
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            df.loc[index, "predicted_answer"] = f"ERROR: {str(e)}"
            df.loc[index, "predicted_answer_en"] = f"ERROR: {str(e)}"
            df.loc[index, "model"] = model_name
            df.loc[index, "system_prompt"] = model.system_prompt
            continue

    # Save results even if some rows had errors
    output_path = os.path.join(os.path.dirname(dataset_path), "predictions")
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(
        os.path.join(
            output_path,
            f"dataset_{language.upper()}_{split}_{model_name}_predicted.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the application in FastAPI or offline mode."
    )
    parser.add_argument(
        "--mode",
        choices=["fastapi", "offline"],
        required=True,
        help="Mode to run the application",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--language",
        choices=["CS", "SK", "UK"],
        required=True,
        help="Dataset language part",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        required=True,
        help="Dataset split to run the prediction",
    )
    parser.add_argument(
        "--model",
        choices=["gemma", "phi4", "aya"],
        required=True,
        help="Model to run the prediction",
    )

    args = parser.parse_args()

    if args.mode == "fastapi":
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.mode == "offline":
        run_offline(args.model, args.language, args.split, args.dataset_path)
