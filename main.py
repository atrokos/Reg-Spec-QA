import argparse
import os
import warnings

import pandas as pd
from tqdm import tqdm

from models.aya_model import AyaModel
from models.gemma_model import GemmaModel
from models.phi4_model import Phi4Model
from models.gemma3qat_model import GemmaQATModel
from utils import load_image_and_encode

model_registry = None  # Only one model can be registered at a time

def run_offline(model_name: str, language: str, split: str, dataset_path: str, batch_size: int = 1):
    print(f"Initializing {model_name} model...")
    model = None
    if model_name == "gemma":
        model = GemmaModel(lang_flag=language)
    elif model_name == "gemmaqat":
        model = GemmaQATModel(lang_flag=language)
    elif model_name == "phi4":
        model = Phi4Model(lang_flag=language)
    elif model_name == "aya":
        model = AyaModel(lang_flag=language)
    else:
        raise ValueError("Invalid model name")

    print(f"Loading dataset for {language} {split} split...")
    df = pd.read_csv(
        os.path.join(dataset_path, f"dataset_{language.upper()}_{split}.csv")
    )
    print(f"Loaded {len(df)} examples")

    assert(batch_size == 1 and "Batch processing disabled for now...")

    print("Processing examples one by one...")
    # Process one by one (original implementation)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image_path = row["image_url"]
            # encoded_image = load_image_and_encode(image_path)
            response = model.generate((image_path, row["question"]))
            df.loc[index, "predicted_answer"] = response

            # Use the same approach for the English question
            response_en = model.generate((image_path, row["question_en"]))
            df.loc[index, "predicted_answer_en"] = response_en
            df.loc[index, "model"] = model_name
            df.loc[index, "system_prompt"] = model.system_prompt
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Save results even if some rows had errors
    output_path = os.path.join(os.path.dirname(dataset_path), "predictions")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(
        output_path,
        f"dataset_{language.upper()}_{split}_{model_name}_predicted.csv",
    )
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    print("Done!")


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
        choices=["gemma", "gemmaqat", "phi4", "aya"],
        required=True,
        help="Model to run the prediction",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the prediction",
    )

    args = parser.parse_args()

    if args.mode == "fastapi":
        raise NotImplementedError("FastAPI mode is not supported on this branch.")
    elif args.mode == "offline":
        run_offline(args.model, args.language, args.split, args.dataset_path, args.batch_size)
