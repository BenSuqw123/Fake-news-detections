import pandas as pd 

# load dataset
df_1 = pd.read_csv("./Data/fever_10000.csv")
df_2 = pd.read_csv("./Data/liar_5000.csv")
df_3 = pd.read_csv("./Data/politifact_data.csv")

# clean data
df_1 = df_1[["claim","label"]]
df_2 = df_2[["claim","label"]]
df_3 = df_3[["text","label"]]

# rename columns
df_3.columns = ["claim","label"]

# change label fever to format

df_1["label"] = df_1["label"].replace({
    "SUPPORTS": "TRUE",
    "REFUTES": "FALSE",
    "NOT ENOUGH INFO": "NEI"
})

df_2["label"] = df_2["label"].replace({
    "true": "TRUE",
    "mostly-true": "TRUE",
    "half-true": "NEI",
    "false": "FALSE",
    "pants-fire": "FALSE",
    "mostly-false": "FALSE",
    "barely-true": "FALSE"
})

df_3["label"] = df_3["label"].replace({
    "true": "TRUE",
    "mostly-true": "TRUE",
    "half-true": "NEI",
    "false": "FALSE",
    "pants-fire": "FALSE",
    "mostly-false": "FALSE",
    "barely-true": "FALSE"
})

# clean text
df_1["claim"] = df_1["claim"].str.strip()
df_2["claim"] = df_2["claim"].str.strip()
df_3["claim"] = df_3["claim"].str.strip()


# combine datasets
df = pd.concat([df_1, df_2, df_3], ignore_index=True)

# drop duplicates
df = df.drop_duplicates(subset=["claim"])

# shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# save dataset
output_path = PROJECT_ROOT / "Transformer" / "combined_data.csv"
df.to_csv(output_path, index=False)
logger.info(f"Combined data saved to {output_path}")
