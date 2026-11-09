import re, pandas as pd, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tr = pd.read_csv("data/train_split.csv")
vec = TfidfVectorizer(analyzer="char", ngram_range=(1,3), max_features=5000)
Xtr = vec.fit_transform(tr["text"])
clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, tr["label"])

text = open("data/raw/analysis_text.txt", encoding="utf-8").read()
text = re.sub(r"\s+", " ", text)
sents = re.split(r"(?<=[。！？])", text)
def clean(s):
    s = s.strip()
    if not re.search(r"[\u4e00-\u9fff]", s): return ""
    s = re.sub(r'^[”’"\'、，。；：]+', '', s)
    s = re.sub(r'[“‘"\'、，；：]+$', '', s)
    return s.strip()
sents = [clean(s) for s in sents]
sents = [s for s in sents if 8 <= len(s) <= 60]

probs = clf.predict_proba(vec.transform(sents))[:,1]
preds = (probs >= 0.5).astype(int)

os.makedirs("output", exist_ok=True)
pd.DataFrame({"text": sents, "pred": preds, "p_psych": probs}).to_csv(
    "output/analysis_predictions.csv", index=False, encoding="utf-8-sig")

n = len(sents)
print(f"共 {n} 句，预测心理 {int(preds.sum())} 句 ({preds.mean():.1%})")
print(pd.DataFrame({"text": sents, "pred": preds, "p_psych": probs.round(2)}).head(15).to_string())
