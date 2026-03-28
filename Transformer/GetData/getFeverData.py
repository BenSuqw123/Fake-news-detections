from datasets import load_dataset
import pandas as pd

dataset = load_dataset("fever", "v1.0")

train = dataset["train"]

df = train.to_pandas()

df = df[["claim","label"]]

sample = df.sample(10000)

sample.to_csv("./Data/fever_10000.csv", index=False)

print("Saved 10000 claims")