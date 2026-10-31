import re, csv, random, os

random.seed(42)
SRC = "data/raw/book.txt"

with open(SRC, encoding="utf-8") as f:
    text = f.read()

# 去 HTML 标签
text = re.sub(r"<[^>]+>", "", text)

# 统一引号、去多余空白
text = text.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
text = re.sub(r"\s+", " ", text)

# 按中文句末标点切句
sentences = re.split(r"(?<=[。！？])", text)

def clean(s):
    s = s.strip().strip('"').strip("'").strip()
    # 去掉句首尾的非中文/非字母数字符号
    s = re.sub(r'^[^\u4e00-\u9fffA-Za-z0-9]+', '', s)
    s = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9。！？]+$', '', s)
    return s.strip()

sentences = [clean(s) for s in sentences]
# 至少含一个中文字，长度 8–60
sentences = [s for s in sentences
             if 8 <= len(s) <= 60 and re.search(r"[\u4e00-\u9fff]", s)]

# 去重
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
