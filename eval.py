import os
import argparse
import pandas as pd
from bert_score import score


def evaluate_metric(
    metric_name: str,
    df: pd.DataFrame,
    target_column: str = "answer",
    predicted_column: str = "predicted_answer",
    model_type: str = "roberta-large_L2",
    lang: str = "en",
) -> pd.DataFrame:
    if metric_name.lower() == "bertscore":
        references = df[target_column].tolist()
        candidates = df[predicted_column].tolist()

        P, R, F1 = score(candidates, references, model_type=model_type, lang=lang)

        df[f"metric_{metric_name}_F1"] = F1.tolist()
        df[f"metric_{metric_name}_P"] = P.tolist()
        df[f"metric_{metric_name}_R"] = R.tolist()
        return df
    else:
        raise ValueError(
            f"Metric '{metric_name}' is not supported. Currently, only 'bertscore' is implemented."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a metric (e.g., BERTScore) on a CSV file with target and predicted columns."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file containing 'target' and 'predicted' columns.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="The name of the metric to compute (e.g., 'bertscore').",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df_result = evaluate_metric(args.metric, df)

    output_path = os.path.join(os.path.dirname(args.input_csv), "eval_results")
    os.makedirs(output_path, exist_ok=True)

    output_csv = os.path.join(
        output_path, f"{os.path.basename(args.input_csv)}_eval.csv"
    )

    df_result.to_csv(output_csv, index=False)

    print(
        f"Evaluation complete. The updated CSV with the '{args.metric}' metric has been saved to '{args.output_csv}'."
    )
