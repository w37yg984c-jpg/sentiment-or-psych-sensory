import pandas as pd, json, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, classification_report)

train = pd.read_csv("data/train_split.csv")
test = pd.read_csv("data/test_split.csv")

vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=5000)
Xtr = vec.fit_transform(train["text"])
Xte = vec.transform(test["text"])

clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, train["label"])
pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)[:, 1]

os.makedirs("output", exist_ok=True)
pd.DataFrame({"text": test["text"], "true": test["label"],
              "pred": pred, "p_psych": proba}).to_csv(
    "output/test_predictions.csv", index=False, encoding="utf-8")

cm = confusion_matrix(test["label"], pred, labels=[0, 1])
p, r, f1, _ = precision_recall_fscore_support(
    test["label"], pred, labels=[0, 1], zero_division=0)

metrics = {
    "accuracy": float(accuracy_score(test["label"], pred)),
    "confusion_matrix": cm.tolist(),
    "labels": [0, 1],
    "precision": p.tolist(),
    "recall": r.tolist(),
    "f1": f1.tolist(),
}
json.dump(metrics, open("output/metrics.json", "w"), ensure_ascii=False, indent=2)

print("Accuracy:", round(metrics["accuracy"], 4))
print("Confusion matrix [[TN,FP],[FN,TP]]:", cm.tolist())
print("Precision:", [round(x,3) for x in p])
print("Recall:", [round(x,3) for x in r])
print("F1:", [round(x,3) for x in f1])
print()
print(classification_report(test["label"], pred,
      target_names=["sensory(0)", "psych(1)"], zero_division=0))
