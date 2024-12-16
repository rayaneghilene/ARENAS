#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split

def main(args):
    # Load data
    df = pd.read_csv(args.data)
    df.rename(columns={'text': 'input', 'class': 'labels'}, inplace=True)
    df = df[["input", "labels"]]

    # Split data and generate prompts
    X_train = []
    X_test = []
    X_eval = []

    for labels in ["offensive", "hate", "abusive", "profane", "severe_toxic", "toxic", "identity_hate", "insult", "obscene", "threat", "aggressive", "neither"]:
        train, test_eval = train_test_split(df[df.labels == labels], train_size=0.8, test_size=0.2, random_state=42)
        test, eval_data = train_test_split(test_eval, test_size=0.5, random_state=42)
        X_train.append(train)
        X_test.append(test)
        X_eval.append(eval_data)

    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_test = pd.concat(X_test).sample(frac=1, random_state=10)
    X_eval = pd.concat(X_eval).sample(frac=1, random_state=10)
    X_train = X_train.reset_index(drop=True)

    def generate_prompt(data_point):
        return f"""
            Categorize the tweet enclosed in square brackets to determine if it is offensive, or hate, or abusive, or profane, or severe_toxic, or toxic, or identity_hate, or insult, or obscene, or threat, or aggressive, or neither,
            and return the answer as the corresponding label:
            "offensive" or "hate" or "abusive" or "profane" or "severe_toxic" or "toxic" or "identity_hate" or "insult" or "obscene" or "threat" or "aggressive" or "neither". 
            [{data_point["input"]}] = {data_point["labels"]}
            """.strip()

    def generate_test_prompt(data_point):
        return f"""
            Categorize the tweet enclosed in square brackets to determine if it is offensive, or hate, or abusive, or profane, or severe_toxic, or toxic, or identity_hate, or insult, or obscene, or threat, or aggressive, or neither, 
            and return the answer as the corresponding label:
            "offensive" or "hate" or "abusive" or "profane" or "severe_toxic" or "toxic" or "identity_hate" or "insult" or "obscene" or "threat" or "aggressive" or "neither".
            [{data_point["input"]}] = """.strip()

    X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=["input"])
    X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), columns=["input"])
    y_true = X_test.labels
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["input"])
    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)

    # Load model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prediction function
    def predict(test, model, tokenizer):
        y_pred = []
        for i in tqdm(range(len(X_test))):
            prompt = X_test.iloc[i]["input"]
            pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=4, temperature=0.00001)
            result = pipe(prompt)
            answer = result[0]['generated_text'].split("=")[-1].strip()
            if "offensive" in answer:
                y_pred.append("offensive")
            elif "hate" in answer:
                y_pred.append("hate")
            elif "abusive" in answer:
                y_pred.append("abusive")
            elif "profane" in answer:
                y_pred.append("profane")
            elif "severe_toxic" in answer:
                y_pred.append("severe_toxic")
            elif "toxic" in answer:
                y_pred.append("toxic")
            elif "identity_hate" in answer:
                y_pred.append("identity_hate")
            elif "insult" in answer:
                y_pred.append("insult")
            elif "obscene" in answer:
                y_pred.append("obscene")
            elif "threat" in answer:
                y_pred.append("threat")
            elif "aggressive" in answer:
                y_pred.append("aggressive")
            else:
                y_pred.append("neither")
        return y_pred

    # LoRA Model Configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training Arguments Configuration
    training_arguments = TrainingArguments(
        output_dir="logs",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, # 4
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2e-5, #2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        evaluation_strategy="epoch"
    )

    # Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="input",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=1024,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained("trained-model")

    # Model Predictions
    y_pred = predict(X_test, model, tokenizer)

    # Evaluate predictions
    evaluate(y_true, y_pred)


def evaluate(y_true, y_pred):
    mapping = {
        "offensive": 11, "hate": 10, "abusive": 9, "profane": 8,
        "severe_toxic": 7, "toxic": 6, "identity_hate": 5, "insult": 4,
        "obscene": 3, "threat": 2, "aggressive": 1, "neither": 0
    }

    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    unique_labels = set(y_true)

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')

    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)

    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    print('\nConfusion Matrix:')
    print(conf_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Llama for SUD")
    parser.add_argument('--data', type=str, help='Path to the dataset')
    args = parser.parse_args()
    main(args)

