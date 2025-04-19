import os
import argparse
import pandas as pd
from models.gemma_model import GemmaModel
from models.phi4_model import Phi4Model
from models.aya_model import AyaModel
from utils import load_image_and_encode

def run_offline(model_name: str, language: str, split: str, dataset_path: str, batch_size: int = 32):
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

    # Process the dataset in batches
    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx + batch_size]
        try:
            image_paths = batch["image_url"].tolist()
            encoded_images = [load_image_and_encode(image_path) for image_path in image_paths]
            questions = batch["question"].tolist()
            questions_en = batch["question_en"].tolist()

            responses = [model.generate((encoded_image, question)) for encoded_image, question in zip(encoded_images, questions)]
            responses_en = [model.generate((encoded_image, question_en)) for encoded_image, question_en in zip(encoded_images, questions_en)]

            df.loc[start_idx:start_idx + batch_size - 1, "predicted_answer"] = responses
            df.loc[start_idx:start_idx + batch_size - 1, "predicted_answer_en"] = responses_en
            df.loc[start_idx:start_idx + batch_size - 1, "model"] = model_name
            df.loc[start_idx:start_idx + batch_size - 1, "system_prompt"] = model.system_prompt
        except Exception as e:
            print(f"Error processing batch starting at index {start_idx}: {e}")
            df.loc[start_idx:start_idx + batch_size - 1, "predicted_answer"] = f"ERROR: {str(e)}"
            df.loc[start_idx:start_idx + batch_size - 1, "predicted_answer_en"] = f"ERROR: {str(e)}"
            df.loc[start_idx:start_idx + batch_size - 1, "model"] = model_name
            df.loc[start_idx:start_idx + batch_size - 1, "system_prompt"] = model.system_prompt
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
        description="Run the application in offline mode."
    )
    parser.add_argument(
        "--mode",
        choices=["offline"],
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
        help="Dataset split to run the prediction on",
    )
    parser.add_argument(
        "--model",
        choices=["gemma", "phi4", "aya"],
        required=True,
        help="Model to run the prediction on",
    )

    args = parser.parse_args()

    if args.mode == "offline":
        run_offline(args.model, args.language, args.split, args.dataset_path)
