import re, csv, random, os

random.seed(42)
SRC = "data/raw/book.txt"

with open(SRC, encoding="utf-8") as f:
    text = f.read()

text = re.sub(r"<[^>]+>", "", text)
text = re.sub(r"\s+", " ", text)

sentences = re.split(r"(?<=[。！？])", text)

def clean(s):
    s = s.strip()
    if not re.search(r"[\u4e00-\u9fff]", s):
        return ""
    s = re.sub(r'^[”’"\'、，。；：]+', '', s)
    s = re.sub(r'[“‘"\'、，；：]+$', '', s)
    return s.strip()

sentences = [clean(s) for s in sentences]
sentences = [s for s in sentences if 8 <= len(s) <= 60]

seen, uniq = set(), []
for s in sentences:
    if s not in seen:
        seen.add(s); uniq.append(s)
sentences = uniq

random.shuffle(sentences)
sentences = sentences[:600]

os.makedirs("data", exist_ok=True)
with open("data/to_label.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "text", "label"])
    for i, s in enumerate(sentences):
        w.writerow([i, s, ""])

print(f"生成 {len(sentences)} 句 -> data/to_label.csv")
