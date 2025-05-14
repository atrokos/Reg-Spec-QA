import os
import sys
import argparse
import pandas as pd
import Levenshtein

from bert_score import score

parser = argparse.ArgumentParser(description="Evaluate a metric (e.g., BERTScore) on a CSV file with target and predicted columns.")
parser.add_argument("--input_csv",type=str, required=True, help="Path to the input CSV file containing 'target' and 'predicted' columns.",)
parser.add_argument("--metric", default="bert_score", type=str, choices=["bert_score", "edit_distance", "counter"], help="The name of the metric to compute (e.g., 'bert_score').",)

def evaluate_bert_score(
    df: pd.DataFrame,
    target_column: str = "answer",
    lang: str = "cz",
):
    # Throw away those rows with ERROR in answer
    df_cleaned = df[~df['predicted_answer'].str.contains("ERROR", na=False)]
    df_cleaned = df_cleaned.dropna(how="any") # predicted used to include some nans 
    split_type = df_cleaned.pop("split_type")
    df_cleaned.insert(0, "split_type", split_type)
    image_url = df_cleaned.pop("image_url")
    df_cleaned.insert(8, "image_url", image_url)

    # Result for predicted_answer (cz question)
    references = df_cleaned[target_column].tolist()
    candidates = df_cleaned["predicted_answer"].tolist()

    P, R, F1 = score(candidates, references, lang=lang)

    df_cleaned.insert(5, f"metric_bert_score_F1_cz", [round(val, 4) for val in F1.tolist()])
    df_cleaned.insert(6, f"metric_bert_score_P_cz", [round(val, 4) for val in P.tolist()])
    df_cleaned.insert(7, f"metric_bert_score_R_cz", [round(val, 4) for val in R.tolist()])

    # Result for predicted_answer_en (question in english)
    candidates = df_cleaned["predicted_answer_en"].tolist()

    P, R, F1 = score(candidates, references, lang=lang)
    
    df_cleaned.insert(9, f"metric_bert_score_F1_en", [round(val, 4) for val in F1.tolist()])
    df_cleaned.insert(10, f"metric_bert_score_P_en", [round(val, 4) for val in P.tolist()])
    df_cleaned.insert(11, f"metric_bert_score_R_en", [round(val, 4) for val in R.tolist()])

    return df_cleaned

def compute_edit_distance(pred, ref, normalized=False):
    pred_str = str(pred)
    ref_str = str(ref)
    
    edit_dist = Levenshtein.distance(pred_str, ref_str)
    
    if normalized:
        return (edit_dist / (len(ref_str) or 1)) * 100
    else:
        return edit_dist

def evaluate_edit_distance(
    df: pd.DataFrame,
    target_column: str = "answer",
):
    df_cleaned = df[~df['predicted_answer'].str.contains("ERROR", na=False)]
    df_cleaned = df_cleaned.dropna(how="any")
    split_type = df_cleaned.pop("split_type")
    df_cleaned.insert(0, "split_type", split_type)
    image_url = df_cleaned.pop("image_url")
    df_cleaned.insert(8, "image_url", image_url)
    
    references = df_cleaned[target_column].tolist()
    candidates_cz = df_cleaned["predicted_answer"].tolist()
    
    edit_dist_abs_cz = [compute_edit_distance(cand, ref, normalized=False) 
                      for cand, ref in zip(candidates_cz, references)]
    edit_dist_pct_cz = [round(compute_edit_distance(cand, ref, normalized=True), 2) 
                      for cand, ref in zip(candidates_cz, references)]
    
    df_cleaned.insert(5, "metric_edit_distance_abs_cz", edit_dist_abs_cz)
    df_cleaned.insert(6, "metric_edit_distance_pct_cz", edit_dist_pct_cz)
    
    candidates_en = df_cleaned["predicted_answer_en"].tolist()
    
    edit_dist_abs_en = [compute_edit_distance(cand, ref, normalized=False) 
                      for cand, ref in zip(candidates_en, references)]
    edit_dist_pct_en = [round(compute_edit_distance(cand, ref, normalized=True), 2) 
                      for cand, ref in zip(candidates_en, references)]
    
    df_cleaned.insert(9, "metric_edit_distance_abs_en", edit_dist_abs_en)
    df_cleaned.insert(10, "metric_edit_distance_pct_en", edit_dist_pct_en)
    
    return df_cleaned

def analyze_answer_frequencies(
    df: pd.DataFrame,
):
    # Očištění dat stejným způsobem jako u jiných metrik
    df_cleaned = df[~df['predicted_answer'].str.contains("ERROR", na=False)]
    df_cleaned = df_cleaned.dropna(how="any")
    
    # Analýza četnosti českých odpovědí
    predictions_cz = df_cleaned["predicted_answer"].tolist()
    freq_cz = {}
    for pred in predictions_cz:
        pred = str(pred).strip()
        if pred in freq_cz:
            freq_cz[pred] += 1
        else:
            freq_cz[pred] = 1
    
    # Řazení podle četnosti (sestupně)
    sorted_freq_cz = sorted(freq_cz.items(), key=lambda x: x[1], reverse=True)
    
    # Analýza četnosti anglických odpovědí
    predictions_en = df_cleaned["predicted_answer_en"].tolist()
    freq_en = {}
    for pred in predictions_en:
        pred = str(pred).strip()
        if pred in freq_en:
            freq_en[pred] += 1
        else:
            freq_en[pred] = 1
    
    # Řazení podle četnosti (sestupně)
    sorted_freq_en = sorted(freq_en.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_freq_cz, sorted_freq_en

def save_frequency_analysis_to_files(freq_cz, freq_en, input_csv_path):
    # Vytvoření adresáře pro výstupy
    output_path = os.path.join(os.path.dirname(input_csv_path), "eval_results")
    os.makedirs(output_path, exist_ok=True)
    
    # Získání základního jména souboru bez přípony
    basename = os.path.basename(input_csv_path).split('.')[0]
    
    # Soubor pro české odpovědi
    output_txt_cz = os.path.join(output_path, f"{basename}_answer_frequencies_cz.txt")
    with open(output_txt_cz, 'w', encoding='utf-8') as f:
        f.write("Odpověď\tPočet\n")
        f.write("-" * 50 + "\n")
        for answer, count in freq_cz:
            f.write(f"{answer}\t{count}\n")
    
    output_txt_en = os.path.join(output_path, f"{basename}_answer_frequencies_en.txt")
    with open(output_txt_en, 'w', encoding='utf-8') as f:
        f.write("Answer\tCount\n")
        f.write("-" * 50 + "\n")
        for answer, count in freq_en:
            f.write(f"{answer}\t{count}\n")
            
    return output_txt_cz, output_txt_en

def evaluate(
    metric_name: str,
    df: pd.DataFrame,
    target_column: str = "answer",
    predicted_column: str = "predicted_answer",
    lang: str = "cz",
) -> pd.DataFrame:
    if metric_name.lower() == "bert_score":
        return evaluate_bert_score(df, target_column=target_column, lang=lang)
    elif metric_name.lower() == "edit_distance":
        return evaluate_edit_distance(df, target_column=target_column)
    elif metric_name.lower() == "counter":
        # Pro counter nevracíme DataFrame, ale jen provedeme analýzu
        freq_cz, freq_en = analyze_answer_frequencies(df)
        output_txt_cz, output_txt_en = save_frequency_analysis_to_files(freq_cz, freq_en, args.input_csv)
        print(f"Frequency analysis results saved to {output_txt_cz} and {output_txt_en}.")
        sys.exit(0)
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
        output_path, f"{os.path.basename(args.input_csv)}_eval_{args.metric}.csv"
    )

    df_result.to_csv(output_csv, index=False)

    print(
        f"Evaluation complete. The updated CSV with the '{args.metric}' metric has been saved to '{output_csv}'."
    )