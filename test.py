import pandas as pd
data = pd.read_csv("sleep_dataset.csv")
print(data.describe())
print("\nSleep duration range:", data["sleep_duration"].min(), "-", data["sleep_duration"].max())