#!/usr/bin/env python
"""
Script for processing dataset text files into DataFrames.

Dataset Structure (each file is a plain text file with one example per line):

For CS language:
    - questions.cs_CS.txt  : Native language questions.
    - questions.en_CS.txt  : English questions.
    - answers.cs_CS.txt    : Native language answers.
    - image_urls.CS.txt    : Image URLs.
    - splits.vis.cs        : Split types ("dev" or "test").

For SK language:
    - questions.sk_SK.txt
    - questions.en_SK.txt
    - answers.sk_SK.txt
    - image_urls.SK.txt
    - splits.vis.sk

For UK language:
    - questions.uk_UK.txt
    - questions.en_UK.txt
    - answers.uk_UK.txt
    - image_urls.UK.txt
    - splits.vis.uk

Each line in the files corresponds to one example.
"""

import os
import urllib.request
import urllib.parse
import tarfile
import pandas as pd
import requests
from tqdm import tqdm

RAW_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "raw_dataset"
)
DATASET_URL = (
    "http://ufallab.ms.mff.cuni.cz/~libovicky/regional_vqa_data_219fjewfj32jf329.tgz"
)


def download_and_extract_dataset():

    if not os.path.exists(RAW_DATASET_DIR):
        os.makedirs(RAW_DATASET_DIR, exist_ok=True)
        tgz_path = os.path.join(RAW_DATASET_DIR, "dataset.tgz")
        print("Downloading dataset from", DATASET_URL)
        urllib.request.urlretrieve(DATASET_URL, tgz_path)
        print("Download complete. Extracting dataset...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=RAW_DATASET_DIR)
        print("Extraction complete.")
        os.remove(tgz_path)
    else:
        print("Raw dataset directory already exists. Skipping download.")


def load_file(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


def download_images(image_url_file: str, lang: str, image_base_dir: str = "data/images") -> list[str]:
    """Downloads all images from given file. Returns list of image paths."""
    image_urls = load_file(image_url_file)
    image_paths = []
    headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/117.0.0.0 Safari/537.36"
    }
    image_base_dir = f"{image_base_dir}_{lang}"
    os.makedirs(image_base_dir, exist_ok=True)
    for url in tqdm(image_urls, desc="Downloading images", unit="image", total=len(image_urls)):
        filename = os.path.basename(url)
        filename = urllib.parse.unquote(filename)
        image_path = os.path.join(image_base_dir, filename)
        if not os.path.exists(image_path):
            try:
                response = requests.get(f"https:{url}", headers=headers, stream=True)
                response.raise_for_status()
                with open(image_path, "wb") as img_file:
                    img_file.write(response.content)
                image_paths.append(os.path.abspath(image_path))
            except requests.exceptions.RequestException as e:
                tqdm.write(f"Error downloading {url}: {e}")
                image_paths.append(None)
        else:
            tqdm.write(f"Image {filename} already exists. Skipping download.")
            image_paths.append(os.path.abspath(image_path))
    return image_paths

def process_language(lang: str) -> pd.DataFrame:
    question_file = os.path.join(
        RAW_DATASET_DIR,
        "regional_vqa_data",
        f"questions.{lang.lower()}_{lang.upper()}.txt",
    )
    question_en_file = os.path.join(
        RAW_DATASET_DIR, "regional_vqa_data", f"questions.en_{lang.upper()}.txt"
    )
    answer_file = os.path.join(
        RAW_DATASET_DIR,
        "regional_vqa_data",
        f"answers.{lang.lower()}_{lang.upper()}.txt",
    )
    image_url_file = os.path.join(
        RAW_DATASET_DIR, "regional_vqa_data", f"image_urls.{lang.upper()}.txt"
    )
    split_file = os.path.join(
        RAW_DATASET_DIR, "regional_vqa_data", f"splits.vis.{lang.lower()}"
    )

    # Ensure that all necessary files exist
    for f in [question_file, question_en_file, answer_file, image_url_file, split_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")

    # Read each file into a list of lines
    questions = load_file(question_file)
    questions_en = load_file(question_en_file)
    answers = load_file(answer_file)
    image_urls = download_images(image_url_file, lang)
    splits = load_file(split_file)

    # Check consistency: all files must have the same number of lines (n examples)
    n = len(questions)
    if not (len(questions_en) == len(answers) == len(image_urls) == len(splits) == n):
        raise ValueError("Mismatch in the number of examples among the files.")

    # Construct a dictionary that maps each column to its data
    data = {
        "question": questions,
        "question_en": questions_en,
        "answer": answers,
        "image_url": image_urls,
        "split_type": splits,
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    download_and_extract_dataset()

    # List of language codes to process
    languages = ["CS", "SK", "UK"]
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "processed_dataset"
    )
    os.makedirs(output_dir, exist_ok=True)

    dataframes = {}

    for lang in languages:
        print(f"Processing language: {lang}")
        df = process_language(lang)
        dataframes[lang] = df
        # Display the first 5 examples for verification
        print(f"First 5 examples for {lang} language:")
        print(df.head(), "\n")

    # Optional: Save each DataFrame to a CSV file for further use.
    for lang, df in dataframes.items():
        unique_splits = df["split_type"].unique()
        for split in unique_splits:
            split_df = df[df["split_type"] == split]
            csv_filename = f"{output_dir}/dataset_{lang}_{split}.csv"
            split_df.to_csv(csv_filename, index=False, encoding="utf-8")
            print(
                f"DataFrame for language {lang} split {split} saved to {csv_filename}"
            )
