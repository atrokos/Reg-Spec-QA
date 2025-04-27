import argparse
import os
import warnings

import pandas as pd
from tqdm import tqdm

from models.aya_model import AyaModel
from models.gemma_model import GemmaModel
from models.phi4_model import Phi4Model
from utils import load_image_and_encode

model_registry = None  # Only one model can be registered at a time



def run_offline(model_name: str, language: str, split: str, dataset_path: str, batch_size: int = 1):
    print(f"Initializing {model_name} model...")
    model = None
    if model_name == "gemma":
        model = GemmaModel(lang_flag=language)
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

    if batch_size > 1:
        warnings.warn('Batch processing is not working well and does not save much time, using one by one processing instead')
        batch_size = 1

    # Process in batches if batch_size > 1
    if batch_size > 1:
        print(f"Processing in batches of size {batch_size}...")
        # Prepare batches
        total_rows = len(df)
        for batch_start in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end]

            try:
                # Prepare batch inputs
                batch_inputs = []
                for _, row in batch_df.iterrows():
                    try:
                        image_path = row["image_url"]
                        # encoded_image = load_image_and_encode(image_path)
                        batch_inputs.append((image_path, row["question"]))
                    except Exception as e:
                        print(f"Error loading image for row {batch_start}: {e}")
                        batch_inputs.append((None, row["question"]))

                # Generate responses for the batch
                responses = model.generate_batch(batch_inputs)

                # Update DataFrame with responses
                for i, response in enumerate(responses):
                    idx = batch_start + i
                    df.loc[idx, "predicted_answer"] = response
                    df.loc[idx, "model"] = model_name
                    df.loc[idx, "system_prompt"] = model.system_prompt

                    # Process English questions for the same batch
                    try:
                        if batch_inputs[i][0] is not None:  # Only if image was loaded successfully
                            response_en = model.generate((batch_inputs[i][0], row["question_en"]))
                            df.loc[idx, "predicted_answer_en"] = response_en
                        else:
                            df.loc[idx, "predicted_answer_en"] = "ERROR: Failed to load image"
                    except Exception as e:
                        print(f"Error processing English question for row {idx}: {e}")
                        df.loc[idx, "predicted_answer_en"] = f"ERROR: {str(e)}"

            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                for idx in range(batch_start, batch_end):
                    df.loc[idx, "predicted_answer"] = f"ERROR: {str(e)}"
                    df.loc[idx, "predicted_answer_en"] = f"ERROR: {str(e)}"
                    df.loc[idx, "model"] = model_name
                    df.loc[idx, "system_prompt"] = model.system_prompt
    else:
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
        choices=["gemma", "phi4", "aya"],
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
