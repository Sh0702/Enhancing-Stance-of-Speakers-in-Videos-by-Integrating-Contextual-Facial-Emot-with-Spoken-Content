import spacy
import pandas as pd
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class MaskedABSA:
    def __init__(self, annotations_path):
        self.repo_name = "Anshul99/masked-stance-model"
        self.tokenizer = T5Tokenizer.from_pretrained(self.repo_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.repo_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.nlp = spacy.load("en_core_web_sm")

        self.annotations_path = annotations_path
        self.stop_words = set(stopwords.words('english'))
        self.exclude_pos = {"DET", "ADP", "AUX", "CCONJ", "PART", "PRON", "SCONJ"}
        self.processed_videos = []

    def mask_sentence(self, sentence):
        doc = self.nlp(sentence)
        masked_sentences = {}
        sentence_tokens = word_tokenize(sentence)

        for i, token in enumerate(doc):
            if (token.pos_ in {"NOUN", "ADJ"} or token.ent_type_) and token.pos_ not in self.exclude_pos:
                for j, word in enumerate(sentence_tokens):
                    if word.lower() == token.text.lower() and word.lower() not in self.stop_words:
                        masked_sentence = sentence_tokens.copy()
                        masked_sentence[j] = "[MASK]"
                        masked_sentences[f"{token.text}_{j}"] = " ".join(masked_sentence)
                        break
        return masked_sentences

    def apply_maskedabsa(self, sentence):
        masked_sentences = self.mask_sentence(sentence)
        absa_results = {}
        for key, masked_sentence in masked_sentences.items():
            input_ids = self.tokenizer(masked_sentence, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(input_ids)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            absa_results[key] = decoded_output
        return absa_results

    def process_csv_file(self, file_path):
        df = pd.read_csv(file_path, encoding="utf-8")

        if df.empty or df['Subtitle'].isna().all():
            print(f"File {file_path} is empty or contains only NaN values. Skipping.")
            return

        print(f"Processing {file_path}...")

        df = df.drop(columns=['masked_absa_prediction'], errors='ignore')
        predictions = []

        for sentence in df['Subtitle']:
            if pd.notna(sentence):
                absa_result = self.apply_maskedabsa(sentence)
                predictions.append(absa_result)
            else:
                predictions.append({})

        df['masked_absa_prediction'] = pd.Series(predictions)
        df.to_csv(file_path, index=False)
        print(f"Updated {file_path} with MaskedABSA predictions.")

    def run_batch_inference(self):
        platforms = ["instagram", "tiktok", "youtube"]
        categories = ["alm", "blm"]

        for platform in platforms:
            for category in categories:
                subfolder_path = os.path.join(self.annotations_path, platform, category)
                if os.path.exists(subfolder_path):
                    print(f"Processing subfolder: {subfolder_path}")
                    for file in os.listdir(subfolder_path):
                        if file.endswith(".csv") and file not in self.processed_videos:
                            file_path = os.path.join(subfolder_path, file)
                            self.process_csv_file(file_path)
                else:
                    print(f"Subfolder {subfolder_path} does not exist. Skipping.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MaskedABSA Model Pipeline")
    parser.add_argument("--annotations_path", type=str, required=True, help="Path to subtitles root directory")
    parser.add_argument("--test_sentence", type=str, help="Optional test sentence to run single prediction")

    args = parser.parse_args()

    absa = MaskedABSA(annotations_path=args.annotations_path)

    if args.test_sentence:
        print(f"Running ABSA on: {args.test_sentence}")
        result = absa.apply_maskedabsa(args.test_sentence)
        for key, val in result.items():
            print(f"{key} => {val}")
    else:
        absa.run_batch_inference()
