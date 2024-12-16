import pandas as pd
from os import listdir

output_df = pd.DataFrame(columns = ['text', 'class'])
datasets = [f for f in listdir("0 Preprocessed Datasets") if (f[-4:] == ".csv" and f[:3] != "all")]

for dataset in datasets:
    df = pd.read_csv(f"0 Preprocessed Datasets/{dataset}")
    output_df = pd.concat([output_df, df])

print(output_df.head)
output_df.to_csv("0 Preprocessed Datasets/all.csv", index = False)