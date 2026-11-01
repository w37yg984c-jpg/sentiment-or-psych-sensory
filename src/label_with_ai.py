import os, csv, time
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"], timeout=60)
MODEL = "z-ai/glm-5.3-flash"

PROMPT = """判断下面这句话属于哪一类，只输出一个数字：
0 = 感官描写：主要描写能被五官感知的事物、身体动作、物理状态。
1 = 心理描写：主要描写人物内心想法、情绪、判断、回忆、意图。
若两者都不明显或难以判断，输出 -1。
只输出 0、1 或 -1，不要解释。

句子：{s}
"""

def label_one(s):
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(s=s)}],
            temperature=0,
        )
        out = r.choices[0].message.content.strip()
        return out if out in {"0", "1", "-1"} else "-1"
    except Exception as e:
        print("ERR", type(e).__name__, str(e)[:100], flush=True)
        return ""

rows = list(csv.DictReader(open("data/to_label.csv", encoding="utf-8")))
for i, row in enumerate(rows):
    if row["label"] in {"0", "1", "-1"}:
        continue
    val = label_one(row["text"])
    if val == "":
        continue
    row["label"] = val
    print(f"{i}/{len(rows)} -> {val}", flush=True)
    if i % 20 == 0:
        with open("data/ai_labeled.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "text", "label"])
            w.writeheader(); w.writerows(rows)
    time.sleep(0.5)

with open("data/ai_labeled.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "text", "label"])
    w.writeheader(); w.writerows(rows)
print("完成，已写 data/ai_labeled.csv")
