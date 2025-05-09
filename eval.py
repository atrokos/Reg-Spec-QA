import os
import argparse
import pandas as pd

from bert_score import score

parser = argparse.ArgumentParser(description="Evaluate a metric (e.g., BERTScore) on a CSV file with target and predicted columns.")
parser.add_argument("--input_csv",type=str, required=True, help="Path to the input CSV file containing 'target' and 'predicted' columns.",)
parser.add_argument("--metric", type=str, required=True, help="The name of the metric to compute (e.g., 'bertscore').",)
parser.add_argument("--summarize", type=str, default=False, required=False, help="",)

def evaluate(
    metric_name: str,
    df: pd.DataFrame,
    target_column: str = "answer",
    predicted_column: str = "predicted_answer",
    model_type: str = "roberta-large_L2",
    lang: str = "cz",
) -> pd.DataFrame:
    if metric_name.lower() == "bert_score":
        # Throw away those rows with ERROR in answer
        df_cleaned = df[~df['predicted_answer'].str.contains("ERROR", na=False)]
        split_type = df_cleaned.pop("split_type")
        df_cleaned.insert(0, "split_type", split_type)
        image_url = df_cleaned.pop("image_url")
        df_cleaned.insert(8, "image_url", image_url)

        # Result for predicted_answer (cz question)
        references = df_cleaned[target_column].tolist()
        candidates = df_cleaned["predicted_answer"].tolist()

        P, R, F1 = score(candidates, references, lang=lang)

        df_cleaned.insert(5, f"metric_{metric_name}_F1_cz", [round(val, 4) for val in F1.tolist()])
        df_cleaned.insert(6, f"metric_{metric_name}_P_cz", [round(val, 4) for val in P.tolist()])
        df_cleaned.insert(7, f"metric_{metric_name}_R_cz", [round(val, 4) for val in R.tolist()])

        # Result for predicted_answer_en (question in english)
        candidates = df_cleaned["predicted_answer_en"].tolist()

        P, R, F1 = score(candidates, references, lang=lang)
        
        df_cleaned.insert(9, f"metric_{metric_name}_F1_en", [round(val, 4) for val in F1.tolist()])
        df_cleaned.insert(10, f"metric_{metric_name}_P_en", [round(val, 4) for val in P.tolist()])
        df_cleaned.insert(11, f"metric_{metric_name}_R_en", [round(val, 4) for val in R.tolist()])

        return df_cleaned
    else:
        raise ValueError(
            f"Metric '{metric_name}' is not supported. Currently, only 'bertscore' is implemented."
        )


if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df_result = evaluate(args.metric, df)

    output_path = os.path.join(os.path.dirname(args.input_csv), "eval_results")
    os.makedirs(output_path, exist_ok=True)

    output_csv = os.path.join(
        output_path, f"{os.path.basename(args.input_csv)}_eval.csv"
    )

    df_result.to_csv(output_csv, index=False)

    print(
        f"Evaluation complete. The updated CSV with the '{args.metric}' metric has been saved to '{output_csv}'."
    )

    if args.summarize:
        pass
