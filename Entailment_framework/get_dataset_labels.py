import pandas as pd
import argparse


def print_unique_labels(file_path):
    label_column='class'
    try:
        df = pd.read_csv(file_path)
        unique_labels = df[label_column].unique()
        print(f"The unique labels in the dataset are: {unique_labels}")
        return unique_labels
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except KeyError:
        print(f"Error: Column '{label_column}' not found in the dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entailment ZSL Experiments")
    parser.add_argument("dataset_path", help="Path to your dataset")

    args = parser.parse_args()
    print_unique_labels(args.dataset_path)