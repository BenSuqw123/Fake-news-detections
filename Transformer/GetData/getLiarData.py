import pandas as pd

# load dataset
train = pd.read_csv("../liar_dataset/train.tsv", sep="\t", header=None)
valid = pd.read_csv("../liar_dataset/valid.tsv", sep="\t", header=None)
test = pd.read_csv("../liar_dataset/test.tsv", sep="\t", header=None)

# merge dataset
df = pd.concat([train, valid, test])

# cột 2 là claim, cột 1 là label
df = df[[1,2]]

df.columns = ["label","claim"]

# sample 3000 rows
sample = df.sample(5000)

sample.to_csv("./Data/liar_5000.csv", index=False)

print("Saved 5000 LIAR claims")