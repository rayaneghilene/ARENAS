import argparse
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def data_preprocess(DATAPATH):
    df= pd.read_csv(DATAPATH)
    sequences_to_classify = list(df['text'])
    true_labels = list(df['class'])
    return sequences_to_classify, true_labels


def return_model_name(name):
    model_dict = {
        'roberta': 'roberta-large-mnli',
        'bart': 'facebook/bart-large-mnli'
    }
    return model_dict.get(name, name)


def get_experiment_stats(true_labels, predicted_labels, candidate_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, labels=candidate_labels, average='macro', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, labels=candidate_labels, average='macro', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, labels=candidate_labels, average='macro', zero_division=1)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Macro F1 Score:", f1)


def entailment_experiment(classifier_name, candidate_labels, DATAPATH, device, hypothesis_template , progress_bar):
    classifier_name = return_model_name(classifier_name)
    classifier = pipeline("zero-shot-classification",
                      model= classifier_name, device=device)
    
    sequences_to_classify, true_labels = data_preprocess(DATAPATH)
    if progress_bar == 'progress-bar-on': 
        predicted_labels = []
        for sequence in tqdm(sequences_to_classify, desc="Classifying sequences"):
            classifier_output = classifier(sequence, candidate_labels, hypothesis_template= hypothesis_template)
            predicted_label = classifier_output['labels'][0]
            predicted_labels.append(predicted_label)

    else:
        classifier_outputs = classifier(sequences_to_classify, candidate_labels, hypothesis_template = hypothesis_template )
        predicted_labels = [output['labels'][0] for output in classifier_outputs]
        

    get_experiment_stats(true_labels, predicted_labels, candidate_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entailment ZSL Experiments")
    parser.add_argument("--candidate_labels", nargs="+", help="List of candidate labels")
    parser.add_argument("--dataset_path", help="Path to the dataset")
    parser.add_argument("--model_name", choices=['roberta', 'bart'], help="Name of the model ('roberta' or 'bart')")
    parser.add_argument("--device", type=int, choices=[0, 1], help="Device (0 or 1)")
    parser.add_argument("--hypothesis_template", help="hypothesis template for the classification")
    parser.add_argument("--progress_bar", nargs="?", const='progress-bar-on', default=None, choices=['progress-bar-on', 'progress-bar-off'], help="Choose 'progress-bar-on' or 'progress-bar-off'. If not provided, 'progress-bar-off' will be used.")
    args = parser.parse_args() 

    entailment_experiment(args.model_name, 
                          args.candidate_labels, 
                          args.dataset_path, 
                          args.device, 
                          args.hypothesis_template,
                          args.progress_bar)