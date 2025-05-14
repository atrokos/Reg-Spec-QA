#!/usr/bin/env python3

import os
import argparse
import pandas as pd

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re


token = os.environ.get("HF_TOKEN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description="Evaluate a LLM judge on a CSV file with target and predicted columns.")
parser.add_argument("--input_csv",type=str, required=True, help="Path to the input CSV file containing 'target' and 'predicted' columns.",)


def create_judge_prompt(question, golden_answer, predicted_answer):
    few_shot_examples = """
        Example 1:
        Question: Jak se jmenuje tato elektrárna?
        Golden Answer: Dalešice
        Predicted Answer: Neznámý
        LLM judge output:
            {{
            'score': 1
            }}

        Example 2:
        Question: Kdo byl autorem tohoto díla?
        Golden Answer: Matěj Čapek-Chod
        Predicted Answer: Čapek
        LLM judge output:
            {
            'score': 4
            }

        Example 3:
        Question: Koho zobrazuje tato socha?
        Golden Answer: Tomáše Holana
        Predicted Answer: Tomáš Holan
        LLM judge output:
            {
            'score': 5
            }

        Example 4:
        Question: Kde je tato budova?
        Golden Answer: Prosek, Praha
        Predicted Answer: Praha
        LLM judge output:
            {
            'score': 3
            }

        Example 5:
        Question: Kde se nachází stavba na obrázku?
        Golden Answer: Domažlice
        Predicted Answer: Nad řekou
        LLM judge output:
            {
            'score': 1
            }

    """
    return f"""<s>[INST] You are an impartial judge evaluating the quality of a predicted answer compared to a golden answer.
        Never explain your answer. Return only score <number> (one integer) in a strict JSON format. Give me only the JSON!
            {{
            'score': <number>
            }}

        Only a single number between 1 and 5 is acceptable. Any other format will be rejected.
        Review the examples below to understand the expected scoring approach, then evaluate the new question.

    {few_shot_examples}

    Now evaluate this new example:

    Question: {question}

    Golden Answer: {golden_answer}

    Predicted Answer: {predicted_answer}

    Rate the predicted answer on a scale from 1 to 5, where:
    1 = Completely incorrect or irrelevant
    5 = Perfect match in content

    Provide your numerical score in JSON format ONLY. Do not give any explanation or other text, just the JSON score!
    If you write anything else but the score in JSON, a kitten will die...
    You are the judge, give me simple JSON score answer.
    [/INST]"""


def get_judgment(question, golden_answer, predicted_answer, tokenizer, model):
    prompt = create_judge_prompt(question, golden_answer, predicted_answer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.4,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the generated response, not including the prompt
    response = response[len(prompt):]
    return response.strip()

    

def evaluate(df):
    df.drop('image_url', axis=1, inplace=True)

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    ).to(device)

    # Process each row in the dataframe with batch processing
    results_cz = []
    batch_size = 16
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_results = []
        
        for _, row in batch.iterrows():
            judgment = get_judgment(row['question'], row['answer'], row['predicted_answer'], tokenizer, model)
            batch_results.append(judgment)
            print(judgment)
        results_cz.extend(batch_results)
    
    results_en = []
    batch_size = 16
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_results = []
        
        for _, row in batch.iterrows():
            judgment = get_judgment(row['question_en'], row['answer'], row['predicted_answer_en'], tokenizer, model)
            batch_results.append(judgment)
            print(judgment)
            print(len(judgment))
        results_en.extend(batch_results)

    df['llm_judge_score'] = [process_llm_output(res) for res in results_cz]
    df['llm_judge_score_en'] = [process_llm_output(res) for res in results_en]
    
    # Reorder reasonably the dataframe
    cols = list(df.columns)
    cols.remove('llm_judge_score')
    cols.remove('llm_judge_score_en')
    cols.insert(3, 'llm_judge_score')
    cols.insert(4, 'llm_judge_score_en')
    df = df[cols]

    return df


def process_llm_output(llm_output):
    score = 1

    numbers = re.findall(r'-?\d+\.?\d*', llm_output)
    numbers = [float(num) for num in numbers]

    if not numbers or len(llm_output) > 100:
        return "<UNK>" # no score given

    score = sum(numbers) / len(numbers)

    return str(score)


if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df_result = evaluate(df)

    df_result.to_csv(args.input_csv[:-4]+"_llm_eval.csv")