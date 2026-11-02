import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/train.csv")
df = df[df["label"].isin([0, 1])].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2,
    stratify=df["label"], random_state=42)

pd.DataFrame({"text": X_train, "label": y_train}).to_csv(
    "data/train_split.csv", index=False, encoding="utf-8")
pd.DataFrame({"text": X_test, "label": y_test}).to_csv(
    "data/test_split.csv", index=False, encoding="utf-8")

print("训练集:", len(X_train), y_train.value_counts().to_dict())
print("测试集:", len(X_test), y_test.value_counts().to_dict())
