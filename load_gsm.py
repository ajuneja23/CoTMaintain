from datasets import load_dataset

# Load the GSM8K dataset
ds = load_dataset("openai/gsm8k", "main")

# Export each split to a separate CSV file
ds["train"].to_csv("gsm8k_train.csv")
ds["test"].to_csv("gsm8k_test.csv")
